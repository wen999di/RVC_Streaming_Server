import torch
import numpy as np
import torch.nn.functional as F
from collections import deque
from rvc_infer import RealtimeRVCInferer
from torchgate import TorchGate
import logging
import time
import math
 
 
def _resample_1d(wav: torch.Tensor, orig_sr: int, new_sr: int) -> torch.Tensor:
    if orig_sr == new_sr:
        return wav
    if wav.numel() == 0:
        return wav
    orig_sr = int(orig_sr)
    new_sr = int(new_sr)
    if orig_sr <= 0 or new_sr <= 0:
        return wav
    x = wav.detach()
    x_np = x.float().cpu().numpy()
    try:
        from scipy import signal
        g = math.gcd(orig_sr, new_sr)
        up = int(new_sr // g)
        down = int(orig_sr // g)
        y_np = signal.resample_poly(x_np, up, down).astype(np.float32, copy=False)
        y = torch.from_numpy(y_np).to(wav.device)
        return y
    except Exception:
        x2 = wav.view(1, 1, -1)
        new_len = int(round(x2.shape[-1] * (float(new_sr) / float(orig_sr))))
        if new_len <= 0:
            return wav[:0]
        y2 = F.interpolate(x2, size=new_len, mode="linear", align_corners=True)
        return y2.view(-1)

class RVCCore:
    def __init__(self, config, device=None):
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.tgt_sr = 16000
        self.sr = 16000
        self.input_sr = 16000
        self.output_sr = 16000
        self.bytes_per_sample = 4
        self._pcm_dtype = np.dtype("<f4")
        self.zc = self.tgt_sr // 100
        self.ns_per_sample = 1_000_000_000 // self.sr
        self.silence_db_threshold = -70.0
        self.silence_gate_atten = 0.0
        self.input_noise_reduce = False
        self.output_noise_reduce = False
        self.noise_reduce_prop_decrease = 0.9
        self.rms_mix_rate = 0.8
        self._torchgate = None
        self.pending_samples = 0
        self._in_segments = deque()
        self._inferer = RealtimeRVCInferer(device=self.device)
        self._last_infer_error_log_s = 0.0
        self.update_config(config)

    def update_config(self, config):
        self.config = config or {}
        block_time = float(self.config.get("block_time", 0.25))
        crossfade_time = float(self.config.get("crossfade_length", 0.05))
        extra_time = float(self.config.get("extra_time", 2.0))
        self.passthrough = bool(self.config.get("passthrough", False))
        self.f0_up_key = int(self.config.get("f0_up_key", 0))
        self.formant_shift = float(self.config.get("formant_shift", 0))
        self.f0_method = self.config.get("f0method", "rmvpe")
        self.model_path = str(self.config.get("model_path", "") or "")
        self.index_path = str(self.config.get("index_path", "") or "")
        self.hubert_path = str(self.config.get("hubert_path", "") or "")
        self.rmvpe_path = str(self.config.get("rmvpe_path", "") or "")
        self.index_rate = float(self.config.get("index_rate", 0.0) or 0.0)
        self.silence_db_threshold = float(self.config.get("silence_db_threshold", -70.0) or -70.0)
        self.silence_gate_atten = float(self.config.get("silence_gate_atten", 0.0) or 0.0)
        if self.silence_gate_atten < 0.0:
            self.silence_gate_atten = 0.0
        if self.silence_gate_atten > 1.0:
            self.silence_gate_atten = 1.0
        self.input_noise_reduce = bool(self.config.get("input_noise_reduce", False))
        self.output_noise_reduce = bool(self.config.get("output_noise_reduce", False))
        self.noise_reduce_prop_decrease = float(self.config.get("noise_reduce_prop_decrease", 0.9) or 0.9)
        if self.noise_reduce_prop_decrease < 0.0:
            self.noise_reduce_prop_decrease = 0.0
        if self.noise_reduce_prop_decrease > 1.0:
            self.noise_reduce_prop_decrease = 1.0
        self.rms_mix_rate = float(self.config.get("rms_mix_rate", 0.8) or 0.8)
        if self.rms_mix_rate < 0.0:
            self.rms_mix_rate = 0.0
        if self.rms_mix_rate > 1.0:
            self.rms_mix_rate = 1.0
        if (self.input_noise_reduce or self.output_noise_reduce) and self.model_path and not self.passthrough:
            if self._torchgate is None or int(getattr(self._torchgate, "sr", 0)) != int(self.sr):
                self._torchgate = TorchGate(
                    sr=int(self.sr),
                    nonstationary=True,
                    prop_decrease=float(self.noise_reduce_prop_decrease),
                    n_fft=int(4 * self.zc),
                    win_length=int(4 * self.zc),
                    hop_length=int(self.zc),
                    freq_mask_smooth_hz=500.0,
                    time_mask_smooth_ms=50.0,
                ).to(self.device)
            else:
                self._torchgate.prop_decrease = float(self.noise_reduce_prop_decrease)
        else:
            self._torchgate = None
        self._inferer.configure(
            model_path=self.model_path,
            index_path=self.index_path,
            index_rate=self.index_rate,
            f0_up_key=self.f0_up_key,
            formant_shift=self.formant_shift,
            hubert_path=self.hubert_path,
            rmvpe_path=self.rmvpe_path,
        )
        self._update_buffer_params(block_time, crossfade_time, extra_time)
        self.pending_samples = 0
        self._in_segments = deque()
 
    def warmup(self):
        return self._inferer.warmup(f0method=self.f0_method)

    def process_frame(self, audio_bytes_f32, ts_start_ns=None):
        if not audio_bytes_f32:
            return b"", None
        if ts_start_ns is None:
            ts_start_ns = 0

        self._push_input_segment(audio_bytes_f32, int(ts_start_ns))

        if self.pending_samples < self.block_frame:
            return b"", None

        block_bytes, block_ts = self._consume_block_bytes(int(self.block_frame))

        indata = np.frombuffer(block_bytes, dtype=self._pcm_dtype).astype(np.float32, copy=False).copy()
        input_chunk = torch.from_numpy(indata).to(self.device, dtype=torch.float32)
        input_chunk = self._apply_silence_gate(input_chunk)
        self._append_input_with_ts(input_chunk, block_ts)
        if self.input_noise_reduce and self._torchgate is not None and self.model_path and not self.passthrough:
            self._apply_input_noise_reduce()

        block_out, out_start_ns = self._process_block_with_ts()
        block_out = torch.nan_to_num(block_out, nan=0.0, posinf=0.0, neginf=0.0).clamp(-1.0, 1.0)
        out_bytes = block_out.detach().to("cpu", dtype=torch.float32).numpy().astype(self._pcm_dtype, copy=False).tobytes()
        return out_bytes, out_start_ns

    def _apply_silence_gate(self, x: torch.Tensor) -> torch.Tensor:
        if x.numel() == 0:
            return x
        if float(self.silence_db_threshold) <= -120.0:
            return x
        zc = int(self.zc)
        if zc <= 0:
            return x
        n = int(x.shape[0])
        frames = n // zc
        if frames <= 0:
            return x
        tail = n - frames * zc
        x_main = x[: frames * zc].view(frames, zc)
        rms = torch.sqrt(torch.mean(x_main * x_main, dim=1) + 1e-12)
        db = 20.0 * torch.log10(rms + 1e-12)
        mask = db < float(self.silence_db_threshold)
        if not torch.any(mask):
            return x
        factors = torch.ones_like(rms)
        factors[mask] = float(self.silence_gate_atten)
        y_main = x_main * factors.view(-1, 1)
        if tail > 0:
            return torch.cat([y_main.view(-1), x[-tail:]], dim=0)
        return y_main.view(-1)

    def _apply_input_noise_reduce(self):
        if self._torchgate is None:
            return
        tail_len = int(self.sola_buffer_frame + self.block_frame)
        if tail_len <= 0:
            return
        if not hasattr(self, "input_wav_denoise") or not hasattr(self, "nr_buffer"):
            return
        if int(self.input_wav.shape[0]) < tail_len:
            return
        try:
            seg = self.input_wav[-tail_len:]
            den = self._torchgate(seg)
            if int(den.shape[0]) < tail_len:
                den = F.pad(den, (0, tail_len - int(den.shape[0])))
            n = int(self.sola_buffer_frame)
            if n > 0:
                den[:n] *= self.fade_in_window
                den[:n] += self.nr_buffer * self.fade_out_window
            self.input_wav_denoise[-int(self.block_frame) :] = den[: int(self.block_frame)]
            self.nr_buffer[:] = den[int(self.block_frame) : int(self.block_frame) + int(self.sola_buffer_frame)]
        except Exception:
            now_s = time.time()
            if now_s - float(self._last_infer_error_log_s or 0.0) > 2.0:
                logging.exception("Input noise reduce failed")
                self._last_infer_error_log_s = now_s
    
    def _rms_env(self, x: torch.Tensor, *, frame_length: int, hop_length: int, out_len: int) -> torch.Tensor:
        if out_len <= 0:
            return x[:0]
        if x.numel() == 0:
            return torch.zeros(out_len, device=x.device, dtype=x.dtype)
        if int(x.shape[0]) < out_len:
            x = F.pad(x, (0, out_len - int(x.shape[0])))
        else:
            x = x[:out_len]
        k = int(frame_length)
        h = int(hop_length)
        if k < 1:
            k = 1
        if h < 1:
            h = 1
        x2 = x.float().view(1, 1, -1)
        pad = k // 2
        rms = torch.sqrt(F.avg_pool1d(F.pad(x2 * x2, (pad, pad)), kernel_size=k, stride=h) + 1e-12)
        rms = F.interpolate(rms, size=out_len, mode="linear", align_corners=True).view(-1)
        return rms.to(dtype=x.dtype)

    def _push_input_segment(self, pcm_bytes, ts_start_ns):
        sample_count = len(pcm_bytes) // self.bytes_per_sample
        if sample_count <= 0:
            return

        self._in_segments.append([memoryview(pcm_bytes), int(ts_start_ns), 0])
        self.pending_samples += sample_count

        max_pending = int(self.block_frame) * 2
        if self.pending_samples > max_pending:
            self._drop_old_samples(self.pending_samples - max_pending)

    def _drop_old_samples(self, drop_samples):
        remaining = int(drop_samples)
        while remaining > 0 and self._in_segments:
            buf, ts, pos = self._in_segments[0]
            avail_samples = (len(buf) - pos) // self.bytes_per_sample
            if avail_samples <= 0:
                self._in_segments.popleft()
                continue
            if avail_samples <= remaining:
                self._in_segments.popleft()
                self.pending_samples -= avail_samples
                remaining -= avail_samples
            else:
                pos += remaining * self.bytes_per_sample
                ts += remaining * self.ns_per_sample
                self._in_segments[0] = [buf, ts, pos]
                self.pending_samples -= remaining
                remaining = 0

    def _consume_block_bytes(self, block_samples):
        need_bytes = int(block_samples) * self.bytes_per_sample
        out = bytearray(need_bytes)
        out_pos = 0
        block_ts = 0

        if self._in_segments:
            block_ts = int(self._in_segments[0][1])

        while need_bytes > 0 and self._in_segments:
            buf, ts, pos = self._in_segments[0]
            avail = len(buf) - pos
            if avail <= 0:
                self._in_segments.popleft()
                continue
            take = avail if avail < need_bytes else need_bytes
            out[out_pos : out_pos + take] = buf[pos : pos + take]
            out_pos += take
            need_bytes -= take
            consumed_samples = take // self.bytes_per_sample
            pos += take
            ts += consumed_samples * self.ns_per_sample
            if pos >= len(buf):
                self._in_segments.popleft()
            else:
                self._in_segments[0] = [buf, ts, pos]

        consumed_total = int(out_pos // self.bytes_per_sample)
        self.pending_samples -= consumed_total
        if self.pending_samples < 0:
            self.pending_samples = 0
        return bytes(out), block_ts

    def _update_buffer_params(self, block_time, crossfade_time, extra_time):
        block_units = int(np.round(block_time * self.tgt_sr / self.zc))
        if block_units < 1:
            block_units = 1
        self.block_frame = block_units * self.zc
 
        crossfade_units = int(np.round(crossfade_time * self.tgt_sr / self.zc))
        if crossfade_units < 1:
            crossfade_units = 1
        self.crossfade_frame = crossfade_units * self.zc
        self.sola_buffer_frame = min(self.crossfade_frame, 4 * self.zc)
        self.sola_search_frame = self.zc
        self.extra_frame = int(np.round(extra_time * self.tgt_sr / self.zc)) * self.zc
        self.skip_head = int(self.extra_frame // self.zc)
        self.return_length = int((self.block_frame + self.sola_buffer_frame + self.sola_search_frame) // self.zc)
        self.block_frame_16k = int(160 * self.block_frame // self.zc)
        self.input_wav = torch.zeros(
            self.extra_frame + self.crossfade_frame + self.sola_search_frame + self.block_frame,
            device=self.device,
            dtype=torch.float32,
        )
        self.input_wav_denoise = self.input_wav.clone()
        self.sola_buffer = torch.zeros(self.sola_buffer_frame, device=self.device, dtype=torch.float32)
        self.nr_buffer = self.sola_buffer.clone()
        self.output_buffer = torch.zeros(
            self.block_frame + self.sola_buffer_frame + self.sola_search_frame,
            device=self.device,
            dtype=torch.float32,
        )
        self.out_nr_buffer = self.sola_buffer.clone()
        self.fade_in_window = (
            torch.sin(
                0.5
                * np.pi
                * torch.linspace(0.0, 1.0, steps=self.sola_buffer_frame, device=self.device, dtype=torch.float32)
            )
            ** 2
        )
        self.fade_out_window = 1 - self.fade_in_window
        self.ts_window = deque([(self.input_wav.shape[0], None)])

    def _append_input_with_ts(self, input_chunk, ts_start_ns):
        if input_chunk.numel() == 0:
            return

        n_samples = input_chunk.shape[0]

        if n_samples >= self.input_wav.shape[0]:
            self.input_wav[:] = input_chunk[-self.input_wav.shape[0] :]
            if hasattr(self, "input_wav_denoise"):
                self.input_wav_denoise[:] = self.input_wav
            self.ts_window.clear()
            offset = n_samples - self.input_wav.shape[0]
            base_ts = ts_start_ns + offset * self.ns_per_sample if ts_start_ns else None
            self.ts_window.append((self.input_wav.shape[0], base_ts))
            return

        shift = n_samples
        self.input_wav[:-shift] = self.input_wav[shift:].clone()
        self.input_wav[-shift:] = input_chunk
        if hasattr(self, "input_wav_denoise"):
            self.input_wav_denoise[:-shift] = self.input_wav_denoise[shift:].clone()
            self.input_wav_denoise[-shift:] = input_chunk

        discard = shift
        while discard > 0 and self.ts_window:
            c, ts = self.ts_window[0]
            if c <= discard:
                self.ts_window.popleft()
                discard -= c
            else:
                new_ts = ts + discard * self.ns_per_sample if ts is not None else None
                self.ts_window[0] = (c - discard, new_ts)
                discard = 0

        self.ts_window.append((n_samples, ts_start_ns))

    def _process_block_with_ts(self):
        infer_wav = None
        wav_for_infer = self.input_wav_denoise if self.input_noise_reduce and hasattr(self, "input_wav_denoise") else self.input_wav
        if self.model_path and not self.passthrough:
            try:
                infer_wav = self._inferer.infer(
                    wav_for_infer,
                    block_frame_16k=self.block_frame_16k,
                    skip_head=self.skip_head,
                    return_length=self.return_length,
                    f0method=self.f0_method,
                )
                info = self._inferer.info
                if info is not None and int(info.tgt_sr) != int(self.sr):
                    infer_wav = _resample_1d(infer_wav, int(info.tgt_sr), int(self.sr))
            except Exception:
                now_s = time.time()
                if now_s - float(self._last_infer_error_log_s or 0.0) > 2.0:
                    logging.exception("RVC infer failed")
                    self._last_infer_error_log_s = now_s
                infer_wav = None
 
        if infer_wav is None:
            infer_wav = wav_for_infer[self.extra_frame :].clone()

        if self.output_noise_reduce and self._torchgate is not None and self.model_path and not self.passthrough:
            try:
                den = self._torchgate(infer_wav)
                if den.shape[0] == infer_wav.shape[0]:
                    infer_wav = den
            except Exception:
                now_s = time.time()
                if now_s - float(self._last_infer_error_log_s or 0.0) > 2.0:
                    logging.exception("Output noise reduce failed")
                    self._last_infer_error_log_s = now_s
        
        if self.model_path and not self.passthrough and float(self.rms_mix_rate) < 1.0:
            try:
                ref = (wav_for_infer if not (self.input_noise_reduce and hasattr(self, "input_wav_denoise")) else self.input_wav_denoise)[
                    self.extra_frame :
                ]
                out_len = int(infer_wav.shape[0])
                rms1 = self._rms_env(ref, frame_length=int(4 * self.zc), hop_length=int(self.zc), out_len=out_len)
                rms2 = self._rms_env(infer_wav, frame_length=int(4 * self.zc), hop_length=int(self.zc), out_len=out_len)
                rms2 = torch.maximum(rms2, torch.zeros_like(rms2) + 1e-3)
                exp = torch.tensor(1.0 - float(self.rms_mix_rate), device=infer_wav.device, dtype=infer_wav.dtype)
                infer_wav = infer_wav * torch.pow(rms1 / rms2, exp)
            except Exception:
                now_s = time.time()
                if now_s - float(self._last_infer_error_log_s or 0.0) > 2.0:
                    logging.exception("RMS mix failed")
                    self._last_infer_error_log_s = now_s

        out_wav, sola_offset = self._sola_logic_with_offset(infer_wav)
        read_idx = self.extra_frame + sola_offset
        out_ts = self._get_ts_at(read_idx)
        return out_wav, out_ts

    def _get_ts_at(self, idx):
        curr = 0
        for c, ts in self.ts_window:
            if curr + c > idx:
                offset = idx - curr
                return ts + offset * self.ns_per_sample if ts is not None else 0
            curr += c
        return 0

    def _sola_logic_with_offset(self, infer_wav):
        min_len = int(self.sola_buffer_frame + self.sola_search_frame)
        if infer_wav.shape[0] < min_len:
            infer_wav = F.pad(infer_wav, (0, min_len - infer_wav.shape[0]))
        conv_input = infer_wav[None, None, : self.sola_buffer_frame + self.sola_search_frame]
        cor_nom = F.conv1d(conv_input, self.sola_buffer[None, None, :])
        cor_den = torch.sqrt(
            F.conv1d(
                conv_input**2,
                torch.ones(1, 1, self.sola_buffer_frame, device=self.device),
            )
            + 1e-8
        )
        sola_offset = torch.argmax(cor_nom[0, 0] / cor_den[0, 0])
        sola_offset_val = int(sola_offset.item())

        infer_wav = infer_wav[sola_offset:]

        if infer_wav.shape[0] < self.block_frame + self.sola_buffer_frame:
            needed = self.block_frame + self.sola_buffer_frame - infer_wav.shape[0]
            infer_wav = F.pad(infer_wav, (0, needed))

        infer_wav[: self.sola_buffer_frame] *= self.fade_in_window
        infer_wav[: self.sola_buffer_frame] += self.sola_buffer * self.fade_out_window

        self.sola_buffer[:] = infer_wav[self.block_frame : self.block_frame + self.sola_buffer_frame]

        out = infer_wav[: self.block_frame]
        if out.shape[0] < self.block_frame:
            out = F.pad(out, (0, self.block_frame - out.shape[0]))

        return out, sola_offset_val
