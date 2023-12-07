import math
import torch
from torch import nn
import torch.nn.functional as F
import numpy as np

from src.model.base_model import BaseModel


# https://github.com/XuMuK1/dla2023/blob/2023/week10/antispoofing_seminar.ipynb


class SincConv_fast(nn.Module):
    """Sinc-based convolution
    Parameters
    ----------
    in_channels : `int`
        Number of input channels. Must be 1.
    out_channels : `int`
        Number of filters.
    kernel_size : `int`
        Filter length.
    sample_rate : `int`, optional
        Sample rate. Defaults to 16000.
    Usage
    -----
    See `torch.nn.Conv1d`
    Reference
    ---------
    Mirco Ravanelli, Yoshua Bengio,
    "Speaker Recognition from raw waveform with SincNet".
    https://arxiv.org/abs/1808.00158
    """

    @staticmethod
    def to_mel(hz):
        return 2595 * np.log10(1 + hz / 700)

    @staticmethod
    def to_hz(mel):
        return 700 * (10 ** (mel / 2595) - 1)

    def __init__(
        self,
        out_channels,
        kernel_size,
        sample_rate=16000,
        in_channels=1,
        stride=1,
        padding=0,
        dilation=1,
        bias=False,
        groups=1,
        min_low_hz=0,
        min_band_hz=0,
    ):
        super().__init__()

        if in_channels != 1:
            # msg = (f'SincConv only support one input channel '
            #       f'(here, in_channels = {in_channels:d}).')
            msg = "SincConv only support one input channel (here, in_channels = {%i})" % (in_channels)
            raise ValueError(msg)

        self.out_channels = out_channels
        self.kernel_size = kernel_size

        # Forcing the filters to be odd (i.e, perfectly symmetrics)
        if kernel_size % 2 == 0:
            self.kernel_size = self.kernel_size + 1

        self.stride = stride
        self.padding = padding
        self.dilation = dilation

        if bias:
            raise ValueError("SincConv does not support bias.")
        if groups > 1:
            raise ValueError("SincConv does not support groups.")

        self.sample_rate = sample_rate
        self.min_low_hz = min_low_hz
        self.min_band_hz = min_band_hz

        # initialize filterbanks such that they are equally spaced in Mel scale
        low_hz = 30
        high_hz = self.sample_rate / 2 - (self.min_low_hz + self.min_band_hz)

        mel = np.linspace(self.to_mel(low_hz), self.to_mel(high_hz), self.out_channels + 1)
        hz = self.to_hz(mel)

        # filter lower frequency (out_channels, 1)
        self.low_hz_ = nn.Parameter(torch.Tensor(hz[:-1]).view(-1, 1))

        # filter frequency band (out_channels, 1)
        self.band_hz_ = nn.Parameter(torch.Tensor(np.diff(hz)).view(-1, 1))

        # Hamming window
        # self.window_ = torch.hamming_window(self.kernel_size)
        n_lin = torch.linspace(0, (self.kernel_size / 2) - 1, steps=int((self.kernel_size / 2)))  # computing only half of the window
        self.window_ = 0.54 - 0.46 * torch.cos(2 * math.pi * n_lin / self.kernel_size)

        # (1, kernel_size/2)
        n = (self.kernel_size - 1) / 2.0
        self.n_ = 2 * math.pi * torch.arange(-n, 0).view(1, -1) / self.sample_rate  # Due to symmetry, I only need half of the time axes

    def forward(self, waveforms):
        """
        Parameters
        ----------
        waveforms : `torch.Tensor` (batch_size, 1, n_samples)
            Batch of waveforms.
        Returns
        -------
        features : `torch.Tensor` (batch_size, out_channels, n_samples_out)
            Batch of sinc filters activations.
        """

        self.n_ = self.n_.to(waveforms.device)

        self.window_ = self.window_.to(waveforms.device)

        low = self.min_low_hz + torch.abs(self.low_hz_)

        high = torch.clamp(low + self.min_band_hz + torch.abs(self.band_hz_), self.min_low_hz, self.sample_rate / 2)
        band = (high - low)[:, 0]

        f_times_t_low = torch.matmul(low, self.n_)
        f_times_t_high = torch.matmul(high, self.n_)

        band_pass_left = ((torch.sin(f_times_t_high) - torch.sin(f_times_t_low)) / (self.n_ / 2)) * self.window_
        # Equivalent of Eq.4 of the reference paper (SPEAKER RECOGNITION FROM RAW WAVEFORM WITH SINCNET).
        # I just have expanded the sinc and simplified the terms. This way I avoid several useless computations.
        band_pass_center = 2 * band.view(-1, 1)
        band_pass_right = torch.flip(band_pass_left, dims=[1])

        band_pass = torch.cat([band_pass_left, band_pass_center, band_pass_right], dim=1)

        band_pass = band_pass / (2 * band[:, None])

        self.filters = (band_pass).view(self.out_channels, 1, self.kernel_size)

        return F.conv1d(waveforms, self.filters, stride=self.stride, padding=self.padding, dilation=self.dilation, bias=None, groups=1)


tmp = 1


class FMS(nn.Module):
    def __init__(self):
        super().__init__()

        self.avgpool = nn.AvgPool1d(1)
        self.attention = nn.Linear(in_features=tmp, out_features=tmp)

    def forward(self, x):
        out = F.adaptive_avg_pool1d(x.view(x.shape[0], -1), 1)
        out = self.attention(out)
        out = F.sigmoid().view(out.shape[0], out.shape[1], -1)
        return x * out + out


class ResBlock(nn.Module):
    def __init__(self, kernel_size):
        super().__init__()

        self.layers = nn.Sequential(
            nn.BatchNorm1d(tmp),
            nn.LeakyReLU(),
            nn.Conv1d(3, 1, kernel_size, padding="same"),
            nn.BatchNorm1d(tmp),
            nn.LeakyReLU(),
            nn.Conv1d(3, 1, kernel_size, padding="same"),
            nn.MaxPool1d(3),
            FMS(),
        )

    def forward(self, x):
        return self.layers(x)


class RawNet2Model(BaseModel):
    def __init__(self, sinc_channels, sinc_filter_length, channels1, channels2):
        super().__init__()

        self.sinc_filters = SincConv_fast(sinc_channels, sinc_filter_length)
        self.resblocks = nn.Sequential(
            *[ResBlock(channels1) for _ in range(2)],
            *[ResBlock(channels2) for _ in range(4)],
        )
        self.grus = nn.Sequential(
            nn.BatchNorm1d(tmp),
            nn.LeakyReLU(),
            nn.GRU(tmp, tmp + 1, 3),
        )
        self.head = nn.Linear(tmp, 2)

    def forward(self, audio, **kwargs):
        print(f"\n{audio.shape=}")
        x = self.sinc_filters(audio)
        print(f"\n{x.shape=}")
        return {"pred": 1}
