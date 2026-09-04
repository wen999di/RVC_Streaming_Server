from __future__ import annotations

import math

import torch
from torch import nn
from torch.nn import functional as F
from torch.nn.utils import spectral_norm, weight_norm

from models import SynthesizerTrnMs768NSFsid


def slice_segments(x: torch.Tensor, starts: torch.Tensor, length: int) -> torch.Tensor:
    output = torch.zeros(
        x.shape[0], x.shape[1], length, device=x.device, dtype=x.dtype
    )
    for row in range(x.shape[0]):
        start = int(starts[row].item())
        output[row] = x[row, :, start : start + length]
    return output


def slice_frames(x: torch.Tensor, starts: torch.Tensor, length: int) -> torch.Tensor:
    output = torch.zeros(x.shape[0], length, device=x.device, dtype=x.dtype)
    for row in range(x.shape[0]):
        start = int(starts[row].item())
        output[row] = x[row, start : start + length]
    return output


def random_segments(
    x: torch.Tensor, lengths: torch.Tensor, segment_frames: int
) -> tuple[torch.Tensor, torch.Tensor]:
    maximum = torch.clamp(lengths - segment_frames + 1, min=1)
    starts = (torch.rand(x.shape[0], device=x.device) * maximum).long()
    return slice_segments(x, starts, segment_frames), starts


class TrainableSynthesizerV2(SynthesizerTrnMs768NSFsid):
    """Training-only forward path for the inference-compatible v2 generator."""

    def forward(
        self,
        phone: torch.Tensor,
        phone_lengths: torch.Tensor,
        pitch: torch.Tensor,
        continuous_f0: torch.Tensor,
        spectrogram: torch.Tensor,
        spectrogram_lengths: torch.Tensor,
        speaker: torch.Tensor,
    ):
        conditioning = self.emb_g(speaker).unsqueeze(-1)
        prior_mean, prior_logs, phone_mask = self.enc_p(
            phone, pitch, phone_lengths
        )
        latent, posterior_mean, posterior_logs, spec_mask = self.enc_q(
            spectrogram, spectrogram_lengths, g=conditioning
        )
        prior_latent = self.flow(latent, spec_mask, g=conditioning)
        latent_slice, starts = random_segments(
            latent, spectrogram_lengths, self.segment_size
        )
        f0_slice = slice_frames(continuous_f0, starts, self.segment_size)
        generated = self.dec(latent_slice, f0_slice, g=conditioning)
        return (
            generated,
            starts,
            phone_mask,
            spec_mask,
            (
                latent,
                prior_latent,
                prior_mean,
                prior_logs,
                posterior_mean,
                posterior_logs,
            ),
        )


class WaveDiscriminator(nn.Module):
    def __init__(self, spectral: bool = False) -> None:
        super().__init__()
        norm = spectral_norm if spectral else weight_norm
        channels = ((1, 16, 15, 1, 1), (16, 64, 41, 4, 4), (64, 256, 41, 4, 16),
                    (256, 1024, 41, 4, 64), (1024, 1024, 41, 4, 256), (1024, 1024, 5, 1, 1))
        self.layers = nn.ModuleList(
            norm(nn.Conv1d(a, b, k, s, groups=g, padding=(k - 1) // 2))
            for a, b, k, s, g in channels
        )
        self.output = norm(nn.Conv1d(1024, 1, 3, 1, padding=1))

    def forward(self, audio: torch.Tensor):
        maps = []
        x = audio
        for layer in self.layers:
            x = F.leaky_relu(layer(x), 0.1)
            maps.append(x)
        x = self.output(x)
        maps.append(x)
        return x.flatten(1), maps


class PeriodDiscriminator(nn.Module):
    def __init__(self, period: int, spectral: bool = False) -> None:
        super().__init__()
        self.period = int(period)
        norm = spectral_norm if spectral else weight_norm
        specs = ((1, 32), (32, 128), (128, 512), (512, 1024), (1024, 1024))
        self.layers = nn.ModuleList()
        for index, (a, b) in enumerate(specs):
            stride = 3 if index < 4 else 1
            self.layers.append(
                norm(nn.Conv2d(a, b, (5, 1), (stride, 1), padding=(2, 0)))
            )
        self.output = norm(nn.Conv2d(1024, 1, (3, 1), padding=(1, 0)))

    def forward(self, audio: torch.Tensor):
        batch, channels, samples = audio.shape
        remainder = samples % self.period
        if remainder:
            audio = F.pad(audio, (0, self.period - remainder), mode="reflect")
            samples = audio.shape[-1]
        x = audio.view(batch, channels, samples // self.period, self.period)
        maps = []
        for layer in self.layers:
            x = F.leaky_relu(layer(x), 0.1)
            maps.append(x)
        x = self.output(x)
        maps.append(x)
        return x.flatten(1), maps


class MultiPeriodDiscriminator(nn.Module):
    def __init__(self, spectral: bool = False) -> None:
        super().__init__()
        self.discriminators = nn.ModuleList(
            [WaveDiscriminator(spectral)]
            + [PeriodDiscriminator(period, spectral) for period in (2, 3, 5, 7, 11, 17, 23, 37)]
        )

    def forward(self, real: torch.Tensor, generated: torch.Tensor):
        real_scores, generated_scores, real_maps, generated_maps = [], [], [], []
        for discriminator in self.discriminators:
            real_score, real_map = discriminator(real)
            generated_score, generated_map = discriminator(generated)
            real_scores.append(real_score)
            generated_scores.append(generated_score)
            real_maps.append(real_map)
            generated_maps.append(generated_map)
        return real_scores, generated_scores, real_maps, generated_maps


def discriminator_loss(real_scores, generated_scores) -> torch.Tensor:
    return sum(
        torch.mean((1.0 - real.float()) ** 2) + torch.mean(generated.float() ** 2)
        for real, generated in zip(real_scores, generated_scores)
    )


def generator_loss(generated_scores) -> torch.Tensor:
    return sum(torch.mean((1.0 - score.float()) ** 2) for score in generated_scores)


def feature_matching_loss(real_maps, generated_maps) -> torch.Tensor:
    total = torch.zeros((), device=generated_maps[0][0].device)
    for real_group, generated_group in zip(real_maps, generated_maps):
        for real, generated in zip(real_group, generated_group):
            total = total + torch.mean(torch.abs(real.detach().float() - generated.float()))
    return total * 2.0


def kl_loss(
    prior_latent: torch.Tensor,
    posterior_logs: torch.Tensor,
    prior_mean: torch.Tensor,
    prior_logs: torch.Tensor,
    mask: torch.Tensor,
) -> torch.Tensor:
    value = prior_logs.float() - posterior_logs.float() - 0.5
    value = value + 0.5 * (prior_latent.float() - prior_mean.float()).pow(2) * torch.exp(
        -2.0 * prior_logs.float()
    )
    denominator = mask.float().sum().clamp_min(1.0)
    return (value * mask.float()).sum() / denominator


_WINDOWS: dict[tuple, torch.Tensor] = {}


def spectrogram(
    audio: torch.Tensor, n_fft: int, hop: int, window_size: int
) -> torch.Tensor:
    key = (window_size, str(audio.device), str(audio.dtype))
    window = _WINDOWS.get(key)
    if window is None:
        window = torch.hann_window(window_size, device=audio.device, dtype=audio.dtype)
        _WINDOWS[key] = window
    padding = max(0, (n_fft - hop) // 2)
    padded = F.pad(audio.unsqueeze(1), (padding, padding), mode="reflect").squeeze(1)
    value = torch.stft(
        padded,
        n_fft=n_fft,
        hop_length=hop,
        win_length=window_size,
        window=window,
        center=False,
        return_complex=True,
    )
    return value.abs().clamp_min(1e-7)


_MEL_BANKS: dict[tuple, torch.Tensor] = {}


def mel_spectrogram_from_linear(
    linear: torch.Tensor,
    *,
    sample_rate: int,
    n_fft: int,
    n_mels: int,
) -> torch.Tensor:
    key = (sample_rate, n_fft, n_mels, str(linear.device), str(linear.dtype))
    bank = _MEL_BANKS.get(key)
    if bank is None:
        import librosa

        values = librosa.filters.mel(
            sr=sample_rate,
            n_fft=n_fft,
            n_mels=n_mels,
            fmin=0.0,
            fmax=None,
        )
        bank = torch.from_numpy(values).to(device=linear.device, dtype=linear.dtype)
        _MEL_BANKS[key] = bank
    return torch.log(torch.matmul(bank, linear).clamp_min(2e-6))


def mel_spectrogram(
    audio: torch.Tensor,
    *,
    sample_rate: int,
    n_fft: int,
    hop: int,
    window_size: int,
    n_mels: int,
) -> torch.Tensor:
    linear = spectrogram(audio, n_fft, hop, window_size)
    return mel_spectrogram_from_linear(
        linear, sample_rate=sample_rate, n_fft=n_fft, n_mels=n_mels
    )
