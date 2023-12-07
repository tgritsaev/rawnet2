import os

import torchaudio
from torch.utils.data import Dataset

from src.utils import DEFAULT_SR


class ASVspoof2019Dataset(Dataset):
    def __init__(self, dir, part, max_sec_length=None, limit=None, **kwargs):
        super().__init__()

        self.dir_w_audio = f"{dir}/LA/LA/ASVspoof2019_LA_{part}/flac"
        self.max_sec_length = max_sec_length

        suffix = "trn" if part == "train" else "trl"
        with open(f"{dir}/LA/LA/ASVspoof2019_LA_cm_protocols/ASVspoof2019.LA.cm.{part}.{suffix}.txt", "r") as fin:
            protocols_txt = fin.readlines()
        self.protocols = []
        for protocol_line in protocols_txt[:limit]:
            protocol_list = protocol_line.split()
            protocol_list[1]
            self.protocols.append([protocol_list[1], 0 if protocol_list[-1] == "spoof" else 1])

    def __len__(self):
        return len(self.protocols)

    def __getitem__(self, idx):
        audio, _ = torchaudio.load(os.path.join(self.dir_w_audio, self.protocols[idx][0]))
        if self.max_sec_length is not None:
            audio = audio[:, : self.max_sec_length * DEFAULT_SR]
        return {"audio": audio, "target": self.protocols[idx][1]}
