import random
from tqdm import tqdm

import PIL
import pandas as pd
import numpy as np
import torch
from torchvision.transforms import ToTensor

from src.trainer.base_trainer import BaseTrainer
from src.logger.utils import plot_spectrogram_to_buf
from src.utils import inf_loop, MetricTracker
from src.utils import DEFAULT_SR


class Trainer(BaseTrainer):
    """
    Trainer class
    """

    def __init__(
        self,
        model,
        criterion,
        metrics,
        optimizer,
        config,
        device,
        dataloaders,
        lr_scheduler=None,
        len_epoch=None,
        skip_oom=True,
    ):
        super().__init__(model, criterion, metrics, optimizer, lr_scheduler, config, device)
        self.skip_oom = skip_oom
        self.config = config
        self.train_dataloader = dataloaders["train"]
        if len_epoch is None:
            # epoch-based training
            self.len_epoch = len(self.train_dataloader)
        else:
            # iteration-based training
            self.train_dataloader = inf_loop(self.train_dataloader)
            self.len_epoch = len_epoch
        self.evaluation_dataloaders = {k: v for k, v in dataloaders.items() if k != "train"}

        self.train_metrics = MetricTracker("loss", "grad_norm", writer=self.writer)
        self.evaluation_metrics = MetricTracker(*[m.name for m in metrics], "grad_norm", writer=self.writer)

    @staticmethod
    def move_batch_to_device(batch, device: torch.device):
        """
        Move all necessary tensors to the GPU
        """
        for tensor_for_gpu in ["audio", "target"]:
            batch[tensor_for_gpu] = batch[tensor_for_gpu].to(device)
        return batch

    def _clip_grad_norm(self):
        if self.config["trainer"].get("grad_norm_clip", None) is not None:
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.config["trainer"]["grad_norm_clip"])

    def _progress(self, batch_idx):
        base = "[{}/{} ({:.0f}%)]"
        if hasattr(self.train_dataloader, "n_samples"):
            current = batch_idx * self.train_dataloader.batch_size
            total = self.train_dataloader.n_samples
        else:
            current = batch_idx
            total = self.len_epoch
        return base.format(current, total, 100.0 * current / total)

    @torch.no_grad()
    def _log_predictions(self, audio, pred, target, examples_to_log=3, **kwargs):
        rows = {}
        convert_to_string = lambda v: "spoof" if v == 0 else "bona-fide"
        for i in range(examples_to_log):
            idx = random.randint(0, audio.shape[0] - 1)
            try_find_i = 0
            #  if i = 0: sample bona-fide, else: sample spoof
            while try_find_i < 10 and ((i == 0 and target[idx] == 1) or (i > 0 and target[idx] == 0)):
                idx = random.randint(0, audio.shape[0] - 1)
                try_find_i += 1
            rows[idx] = {
                "audio": self.writer.wandb.Audio(audio[idx].cpu().squeeze().numpy(), sample_rate=DEFAULT_SR),
                "pred": str(pred[idx].tolist()),
                "target": convert_to_string(target[idx]),
            }

        self.writer.add_table("logs", pd.DataFrame.from_dict(rows, orient="index"))

    def _log_spectrogram(self, spectrogram_batch):
        spectrogram = random.choice(spectrogram_batch.cpu())
        image = PIL.Image.open(plot_spectrogram_to_buf(spectrogram))
        self.writer.add_image("spectrogram", ToTensor()(image))

    @torch.no_grad()
    def get_grad_norm(self, norm_type=2):
        parameters = self.model.parameters()
        if isinstance(parameters, torch.Tensor):
            parameters = [parameters]
        parameters = [p for p in parameters if p.grad is not None]

        total_norm = torch.norm(
            torch.stack([torch.norm(torch.nan_to_num(p.grad.detach(), nan=0), norm_type).cpu() for p in parameters]),
            norm_type,
        )
        return total_norm.item()

    def _log_scalars(self, metric_tracker: MetricTracker):
        if self.writer is None:
            return
        for metric_name in metric_tracker.keys():
            self.writer.add_scalar(f"{metric_name}", metric_tracker.avg(metric_name))

    def process_batch(self, batch, is_train: bool, batch_idx: int, metrics: MetricTracker):
        batch = self.move_batch_to_device(batch, self.device)
        outputs = self.model(**batch)
        batch.update(outputs)
        if is_train:
            loss = self.criterion(**batch)
            batch.update(loss)
            batch["loss"].backward()

            self._clip_grad_norm()
            self.optimizer.step()
            metrics.update("grad_norm", self.get_grad_norm())
            self.optimizer.zero_grad()

            if self.lr_scheduler is not None:
                self.lr_scheduler.step()
            metrics.update("loss", batch["loss"].item())

        return batch

    def _evaluation_epoch(self, epoch, part, dataloader):
        """
        Validate after training an epoch

        :param epoch: Integer, current training epoch.
        :return: A log that contains information about validation
        """
        self.model.eval()
        self.evaluation_metrics.reset()
        all_targets = []
        all_preds = []
        with torch.no_grad():
            for _, batch in tqdm(enumerate(dataloader), desc=part, total=len(dataloader)):
                batch = self.process_batch(batch, False, 0, metrics=self.evaluation_metrics)
                all_targets += batch["target"].detach().cpu().tolist()
                all_preds += (batch["pred"].detach().cpu()[:, 1]).tolist()
            self.writer.set_step(epoch * self.len_epoch, part)
            self._log_predictions(**batch)
            # self._log_spectrogram(batch["spectrogram"])
            self._log_scalars(self.evaluation_metrics)
        for metric in self.metrics:
            self.evaluation_metrics.update(metric.name, metric(np.array(all_targets), np.array(all_preds)))
        return self.evaluation_metrics.result()

    def _train_epoch(self, epoch):
        """
        Training logic for an epoch

        :param epoch: Integer, current training epoch.
        :return: A log that contains average loss and metric in this epoch.
        """
        self.model.train()
        self.train_metrics.reset()
        self.writer.add_scalar("epoch", epoch)
        bar = tqdm(range(self.len_epoch), desc="train")
        for batch_idx, batch in enumerate(self.train_dataloader):
            try:
                batch = self.process_batch(batch, True, batch_idx, metrics=self.train_metrics)
            except RuntimeError as e:
                if "out of memory" in str(e) and self.skip_oom:
                    self.logger.warning("OOM on batch. Skipping batch.")
                    for p in self.model.parameters():
                        if p.grad is not None:
                            del p.grad  # free some memory
                    torch.cuda.empty_cache()
                    continue
                else:
                    raise e
            if batch_idx % self.log_step == 0:
                self.writer.set_step((epoch - 1) * self.len_epoch + batch_idx)
                self.logger.debug("Train Epoch: {} {} Loss: {:.6f}".format(epoch, self._progress(batch_idx), batch["loss"].item()))
                self.writer.add_scalar("learning rate", self.lr_scheduler.get_last_lr()[0])
                self._log_scalars(self.train_metrics)
                self._log_predictions(**batch)
                # we don't want to reset train metrics at the start of every epoch
                # because we are interested in recent train metrics
                last_train_metrics = self.train_metrics.result()
                self.train_metrics.reset()
                bar.update(self.log_step)

            if batch_idx + 1 >= self.len_epoch:
                break

        log = last_train_metrics

        for part, dataloader in self.evaluation_dataloaders.items():
            val_log = self._evaluation_epoch(epoch, part, dataloader)
            log.update(**{f"{part}_{name}": value for name, value in val_log.items()})

        return log
