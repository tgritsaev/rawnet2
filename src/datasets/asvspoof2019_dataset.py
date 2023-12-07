import os

import torchaudio
from torch.utils.data import Dataset


class ASVspoof2019Dataset(Dataset):
    def __init__(self, dir, part, max_length=None, limit=None, **kwargs):
        super().__init__()

        self.max_length = max_length

        with open(f"{dir}/LA/LA/ASVspoof2019_LA_cm_protocols/ASVspoof2019.LA.cm.{part}.trl.txt", "r") as fin:
            protocols_txt = fin.readlines()
        self.protocols = dict()
        for protocol_line in protocols_txt:
            protocol_list = protocol_line.split()
            self.protocols[protocol_list[1]] = 0 if protocol_list[-1] == "spoof" else 1

        self.audio_paths = os.listdir(f"{dir}/LA/LA/ASVspoof2019_LA_{part}/flac")[:limit]

    def __len__(self):
        return len(self.audio_paths)

    def __getitem__(self, idx):
        audio_name = os.path.basename(self.audio_paths[idx])[:-5]
        audio, _ = torchaudio.load(self.audio_paths[idx])
        return {"audio": audio[:, : self.max_length], "target": self.protocols[audio_name]}
