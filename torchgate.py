import torch
from torch.nn import functional as F


def _amp_to_db(x: torch.Tensor) -> torch.Tensor:
    return 20.0 * torch.log10(x.abs().clamp_min(1e-12))


def _temperature_sigmoid(x: torch.Tensor, threshold: float, temp: float) -> torch.Tensor:
    t = float(temp)
    if t <= 0.0:
        t = 1e-6
    return torch.sigmoid((x - float(threshold)) / t)


class TorchGate(torch.nn.Module):
    def __init__(
        self,
        *,
        sr: int,
        nonstationary: bool = True,
        n_std_thresh_stationary: float = 1.5,
        n_thresh_nonstationary: float = 1.3,
        temp_coeff_nonstationary: float = 0.1,
        n_movemean_nonstationary: int = 21,
        prop_decrease: float = 0.9,
        n_fft: int = 640,
        win_length: int | None = None,
        hop_length: int | None = None,
        freq_mask_smooth_hz: float | None = 500.0,
        time_mask_smooth_ms: float | None = 50.0,
    ):
        super().__init__()
        self.sr = int(sr)
        self.nonstationary = bool(nonstationary)
        self.n_std_thresh_stationary = float(n_std_thresh_stationary)
        self.n_thresh_nonstationary = float(n_thresh_nonstationary)
        self.temp_coeff_nonstationary = float(temp_coeff_nonstationary)
        self.n_movemean_nonstationary = int(n_movemean_nonstationary)
        self.prop_decrease = float(prop_decrease)
        if self.prop_decrease < 0.0:
            self.prop_decrease = 0.0
        if self.prop_decrease > 1.0:
            self.prop_decrease = 1.0

        self.n_fft = int(n_fft)
        self.win_length = int(self.n_fft if win_length is None else win_length)
        self.hop_length = int(self.win_length // 4 if hop_length is None else hop_length)
        self.freq_mask_smooth_hz = None if freq_mask_smooth_hz is None else float(freq_mask_smooth_hz)
        self.time_mask_smooth_ms = None if time_mask_smooth_ms is None else float(time_mask_smooth_ms)
        self.register_buffer("window", torch.hann_window(self.win_length, dtype=torch.float32), persistent=False)
        n_k = int(self.n_movemean_nonstationary)
        if n_k < 1:
            n_k = 1
        self.register_buffer("movemean_kernel", torch.ones(1, 1, n_k, dtype=torch.float32), persistent=False)
        self.register_buffer("smoothing_filter", self._make_smoothing_filter(), persistent=False)

    @torch.no_grad()
    def _make_smoothing_filter(self) -> torch.Tensor | None:
        if self.freq_mask_smooth_hz is None and self.time_mask_smooth_ms is None:
            return None
        if self.sr <= 0 or self.n_fft <= 0:
            return None

        n_grad_freq = 1
        if self.freq_mask_smooth_hz is not None:
            hz_per_bin = float(self.sr) / float(self.n_fft / 2.0)
            n_grad_freq = max(1, int(round(float(self.freq_mask_smooth_hz) / hz_per_bin)))

        n_grad_time = 1
        if self.time_mask_smooth_ms is not None and self.hop_length > 0:
            ms_per_frame = (float(self.hop_length) / float(self.sr)) * 1000.0
            n_grad_time = max(1, int(round(float(self.time_mask_smooth_ms) / ms_per_frame)))

        if n_grad_freq == 1 and n_grad_time == 1:
            return None

        k_f = 2 * n_grad_freq + 1
        k_t = 2 * n_grad_time + 1
        filt = torch.ones(1, 1, k_f, k_t, dtype=torch.float32)
        filt /= filt.sum()
        return filt

    @torch.no_grad()
    def _stationary_mask(self, X_db: torch.Tensor) -> torch.Tensor:
        mean = X_db.mean(dim=2, keepdim=True)
        std = X_db.std(dim=2, keepdim=True)
        thresh = mean + std * float(self.n_std_thresh_stationary)
        return X_db > thresh

    @torch.no_grad()
    def _nonstationary_mask(self, X_abs: torch.Tensor) -> torch.Tensor:
        if X_abs.dim() != 3:
            raise ValueError("X_abs must have shape (B, F, T)")
        b, f, t = int(X_abs.shape[0]), int(X_abs.shape[1]), int(X_abs.shape[2])
        kernel = self.movemean_kernel.to(device=X_abs.device, dtype=X_abs.dtype)
        X_flat = X_abs.reshape(b * f, 1, t)
        X_smoothed = (F.conv1d(X_flat, kernel, padding="same") / float(kernel.shape[-1])).view(b, f, t)
        ratio = (X_abs - X_smoothed) / (X_smoothed + 1e-6)
        return _temperature_sigmoid(ratio, float(self.n_thresh_nonstationary), float(self.temp_coeff_nonstationary))

    @torch.no_grad()
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.numel() == 0:
            return x
        if x.dim() == 1:
            x = x.unsqueeze(0)
        if x.dim() != 2:
            raise ValueError("TorchGate expects shape (B, T) or (T,)")

        b, t = int(x.shape[0]), int(x.shape[1])
        win = self.window.to(device=x.device, dtype=torch.float32)
        X = torch.stft(
            x.float(),
            n_fft=self.n_fft,
            hop_length=self.hop_length,
            win_length=self.win_length,
            center=True,
            pad_mode="constant",
            window=win,
            return_complex=True,
        )
        X = X.to(dtype=torch.complex64)
        if self.nonstationary:
            sig_mask = self._nonstationary_mask(X.abs())
        else:
            sig_mask = self._stationary_mask(_amp_to_db(X))

        sig_mask = float(self.prop_decrease) * (sig_mask.to(dtype=torch.float32) - 1.0) + 1.0
        if self.smoothing_filter is not None:
            sig_mask = F.conv2d(sig_mask.unsqueeze(1), self.smoothing_filter.to(sig_mask.dtype), padding="same").squeeze(1)

        Y = X * sig_mask.to(dtype=torch.complex64)
        y = torch.istft(
            Y,
            n_fft=self.n_fft,
            hop_length=self.hop_length,
            win_length=self.win_length,
            center=True,
            window=win,
            length=t,
        )
        y = y.to(dtype=x.dtype)
        if b == 1:
            return y[0]
        return y
