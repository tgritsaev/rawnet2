import argparse
import json
import os
from pathlib import Path

import torch
import torchaudio
from tqdm import tqdm

import src.model as module_model
from src.utils import DEFAULT_SR
from src.utils.parse_config import ConfigParser


def main(config, args):
    logger = config.get_logger("test")

    # define cpu or gpu if possible
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"device: {device}")

    # build model architecture
    model = config.init_obj(config["arch"], module_model)
    logger.info(model)
    logger.info("Loading checkpoint...")
    checkpoint = torch.load(args.checkpoint, map_location=device)
    state_dict = checkpoint["state_dict"]
    if config["n_gpu"] > 1:
        model = torch.nn.DataParallel(model)
    model.load_state_dict(state_dict)
    logger.info("Checkpoint has been loaded.")
    model = model.to(device)
    model.eval()

    cut_length = config["data"]["train"]["datasets"][0]["args"]["max_sec_length"] * DEFAULT_SR

    for audio_name in sorted(os.listdir(args.input_dir)):
        audio, sr = torchaudio.load(f"{args.input_dir}/{audio_name}")
        resampled_audio = torchaudio.functional.resample(audio, orig_freq=sr, new_freq=DEFAULT_SR)
        pred = model(resampled_audio[:cut_length])["pred"][0].detach().cpu().numpy()
        verdict = "bona-fide" if abs(pred[0]) > abs(pred[1]) else "spoofed"
        logger.info(f"{audio_name}:\nverdict: {verdict}\t\tpredictions: {pred}\n")

    logger.info("Testing has ended.")


if __name__ == "__main__":
    args = argparse.ArgumentParser()
    args.add_argument(
        "-c",
        "--config",
        default="configs/train_kaggle.json",
        type=str,
        help="Config path used during training (default: configs/train_kaggle.json).",
    )
    args.add_argument(
        "-cp",
        "--checkpoint",
        default="test_model/as-checkpoint.pth",
        type=str,
        help="Checkpoint path (default: test_model/as-checkpoint.pth).",
    )
    args.add_argument(
        "-i",
        "--input-dir",
        default="test_model/audios",
        type=str,
        help="Directory with input audios (default: test_model/audios).",
    )
    args = args.parse_args()

    model_config = Path(args.config)
    with model_config.open() as fin:
        config = ConfigParser(json.load(fin))

    main(config, args)
