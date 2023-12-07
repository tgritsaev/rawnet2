from typing import List

import torch
import torch.nn.functional as F


def pad_1D_tensor(inputs):
    def pad_data(x, length):
        x_padded = F.pad(x, (0, length - x.shape[0]))
        return x_padded

    max_len = max((len(x) for x in inputs))
    padded = torch.stack([pad_data(x, max_len) for x in inputs])

    return padded


def collate_fn(batch: List[dict]):
    audio = pad_1D_tensor([item["audio"].squeeze(0) for item in batch])
    target = torch.Tensor([item["target"] for item in batch]).to(torch.int32)

    return {"audio": audio, "target": target}
