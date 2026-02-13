import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Optional
import numpy as np
import torch
import torch.nn.functional as F
 
 
class RVCDependencyError(RuntimeError):
    pass
 
 
def _ensure_webui_importable() -> None:
    pass
 
 
def _resample_1d(wav: torch.Tensor, orig_sr: int, new_sr: int) -> torch.Tensor:
    if orig_sr == new_sr:
        return wav
    if wav.numel() == 0:
        return wav
    x = wav.view(1, 1, -1)
    new_len = int(round(x.shape[-1] * (float(new_sr) / float(orig_sr))))
    if new_len <= 0:
        return wav[:0]
    y = F.interpolate(x, size=new_len, mode="linear", align_corners=True)
    return y.view(-1)
 
 
def _resample_units(wav_bt: torch.Tensor, orig_units: int, new_units: int) -> torch.Tensor:
    if orig_units == new_units:
        return wav_bt
    if wav_bt.numel() == 0:
        return wav_bt
    x = wav_bt.unsqueeze(1)
    new_len = int(round(x.shape[-1] * (float(new_units) / float(orig_units))))
    if new_len <= 0:
        return wav_bt[:, :0]
    y = F.interpolate(x, size=new_len, mode="linear", align_corners=True)
    return y.squeeze(1)
 
 
@dataclass(frozen=True)
class LoadedModelInfo:
    tgt_sr: int
    if_f0: int
    version: str
 
 
_HUBERT_CACHE: dict[tuple[str, bool, str], torch.nn.Module] = {}
 
 
def _load_hubert(device: torch.device, is_half: bool, hubert_path: str) -> torch.nn.Module:
    key = (str(device), bool(is_half), str(hubert_path))
    cached = _HUBERT_CACHE.get(key)
    if cached is not None:
        return cached
 
    import fairseq
    models, _, _ = fairseq.checkpoint_utils.load_model_ensemble_and_task(
        [hubert_path],
        suffix="",
    )
    hubert = models[0].to(device)
    hubert = hubert.half() if is_half else hubert.float()
    hubert.eval()
    _HUBERT_CACHE[key] = hubert
    return hubert
 
 
class RealtimeRVCInferer:
    def __init__(
        self,
        *,
        device: Optional[torch.device] = None,
        is_half: Optional[bool] = None,
    ) -> None:
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.is_half = bool(is_half) if is_half is not None else (self.device.type == "cuda")
 
        self._model_path: str = ""
        self._index_path: str = ""
        self._index_rate: float = 0.0
        self._hubert_path: str = ""
        self._rmvpe_path: str = ""

        self._hubert: Optional[torch.nn.Module] = None
        self._net_g: Optional[torch.nn.Module] = None
        self._info: Optional[LoadedModelInfo] = None
 
        self._faiss_index = None
        self._faiss_big_npy = None
 
        self.f0_up_key: int = 0
        self.formant_shift: float = 0.0
 
        self._f0_min = 50.0
        self._f0_max = 1100.0
        self._f0_mel_min = 1127.0 * np.log(1 + self._f0_min / 700.0)
        self._f0_mel_max = 1127.0 * np.log(1 + self._f0_max / 700.0)
 
        self._cache_pitch = torch.zeros(1024, device=self.device, dtype=torch.long)
        self._cache_pitchf = torch.zeros(1024, device=self.device, dtype=torch.float32)
 
        self._rmvpe = None
 
    @property
    def info(self) -> Optional[LoadedModelInfo]:
        return self._info
 
    def configure(
        self,
        *,
        model_path: str,
        index_path: str = "",
        index_rate: float = 0.0,
        f0_up_key: int = 0,
        formant_shift: float = 0.0,
        hubert_path: str = "",
        rmvpe_path: str = "",
    ) -> None:
        self.f0_up_key = int(f0_up_key or 0)
        self.formant_shift = float(formant_shift or 0.0)

        model_path = str(model_path or "")
        index_path = str(index_path or "")
        hubert_path = str(hubert_path or "")
        rmvpe_path = str(rmvpe_path or "")
        index_rate = float(index_rate or 0.0)
        if index_rate < 0.0:
            index_rate = 0.0
        if index_rate > 1.0:
            index_rate = 1.0

        if model_path != self._model_path:
            self._model_path = model_path
            self._net_g = None
            self._info = None
            self._cache_pitch.zero_()
            self._cache_pitchf.zero_()

        if hubert_path != self._hubert_path:
            self._hubert_path = hubert_path
            self._hubert = None  # Reload required

        if rmvpe_path != self._rmvpe_path:
            self._rmvpe_path = rmvpe_path
            self._rmvpe = None  # Reload required

        if index_path != self._index_path or index_rate != self._index_rate:
            self._index_path = index_path
            self._index_rate = index_rate
            self._faiss_index = None
            self._faiss_big_npy = None
 
    def warmup(self, f0method: str = "rmvpe") -> LoadedModelInfo:
        self._ensure_models_loaded()
        assert self._info is not None

        # Perform a dummy inference to warm up the GPU/model
        try:
            # Create a small dummy input (~0.5s silence)
            dummy_wav_len = 8000
            dummy_wav = torch.zeros(dummy_wav_len, dtype=torch.float32)
            
            # Use safe default parameters for warmup
            # block_frame_16k=4096 is a typical chunk size (~250ms)
            dummy_block_frame = 4096
            
            self.infer(
                input_wav_16k=dummy_wav, 
                block_frame_16k=dummy_block_frame,
                skip_head=0,
                return_length=dummy_block_frame,
                f0method=f0method
            )
            
            # Ensure CUDA operations are finished
            if self.device.type == "cuda":
                torch.cuda.synchronize(self.device)
                
        except Exception as e:
            # If warmup fails, propagate the error so the server can report it
            raise e

        return self._info
 
    def _ensure_models_loaded(self) -> None:
        if not self._model_path:
            raise RuntimeError("缺少 model_path")

        if self._hubert is None:
            hubert_path = self._hubert_path
            # 如果未指定 hubert_path，尝试默认路径
            if not hubert_path:
                files_dir = Path(__file__).parent / "files"
                alt_hubert = files_dir / "hubert_base.pt"
                if alt_hubert.exists():
                    hubert_path = str(alt_hubert)
            
            if not os.path.exists(hubert_path):
                 # 如果仍然找不到，尝试在 server/files 下找传入的文件名(如果是纯文件名)
                 if self._hubert_path and not os.path.isabs(self._hubert_path):
                     files_dir = Path(__file__).parent / "files"
                     alt = files_dir / self._hubert_path
                     if alt.exists():
                         hubert_path = str(alt)
            
            if not os.path.exists(hubert_path):
                 raise FileNotFoundError(f"找不到 HuBERT 权重：{hubert_path}")

            self._hubert = _load_hubert(self.device, self.is_half, hubert_path)

        if self._net_g is None or self._info is None:
            if not os.path.exists(self._model_path):
                raise FileNotFoundError(f"找不到音色模型：{self._model_path}")

            # 内联模型加载逻辑，不依赖 infer.lib.jit.get_synthesizer
            from models import (
                SynthesizerTrnMs256NSFsid,
                SynthesizerTrnMs256NSFsid_nono,
                SynthesizerTrnMs768NSFsid,
                SynthesizerTrnMs768NSFsid_nono,
            )

            load_map_location = self.device if self.device.type == "cuda" else "cpu"
            try:
                try:
                    cpt = torch.load(self._model_path, map_location=load_map_location, weights_only=False)
                except TypeError:
                    cpt = torch.load(self._model_path, map_location=load_map_location)
            except RuntimeError as e:
                if self.device.type == "cuda" and "out of memory" in str(e).lower():
                    try:
                        cpt = torch.load(self._model_path, map_location="cpu", weights_only=False)
                    except TypeError:
                        cpt = torch.load(self._model_path, map_location="cpu")
                else:
                    raise
            tgt_sr = cpt["config"][-1]
            cpt["config"][-3] = cpt["weight"]["emb_g.weight"].shape[0]
            if_f0 = cpt.get("f0", 1)
            version = cpt.get("version", "v1")

            if version == "v1":
                if if_f0 == 1:
                    net_g = SynthesizerTrnMs256NSFsid(*cpt["config"], is_half=self.is_half)
                else:
                    net_g = SynthesizerTrnMs256NSFsid_nono(*cpt["config"])
            elif version == "v2":
                if if_f0 == 1:
                    net_g = SynthesizerTrnMs768NSFsid(*cpt["config"], is_half=self.is_half)
                else:
                    net_g = SynthesizerTrnMs768NSFsid_nono(*cpt["config"])
            
            del net_g.enc_q
            net_g = net_g.to(self.device)
            net_g = net_g.half() if self.is_half else net_g.float()
            net_g.load_state_dict(cpt["weight"], strict=False)
            net_g.eval()
            
            self._net_g = net_g
            self._info = LoadedModelInfo(tgt_sr=tgt_sr, if_f0=if_f0, version=version)
 
        if self._index_rate > 0.0 and self._index_path:
            if self._faiss_index is None:
                import faiss
                if not os.path.exists(self._index_path):
                    raise FileNotFoundError(f"找不到 index：{self._index_path}")
                index = faiss.read_index(self._index_path)
                big_npy = index.reconstruct_n(0, index.ntotal)
                
                if self.device.type == "cuda":
                    try:
                        res = faiss.StandardGpuResources()
                        index = faiss.index_cpu_to_gpu(res, 0, index)
                    except Exception as e:
                        print(f"Failed to move faiss index to GPU: {e}")

                self._faiss_index = index
                self._faiss_big_npy = big_npy
                if self.device.type == "cuda" and isinstance(self._faiss_big_npy, np.ndarray):
                    self._faiss_big_npy = torch.from_numpy(self._faiss_big_npy).to(self.device)

    def _get_f0_post(self, f0) -> tuple[torch.Tensor, torch.Tensor]:
        if not torch.is_tensor(f0):
            f0 = torch.from_numpy(f0)
        f0 = f0.float().to(self.device).squeeze()
        f0_mel = 1127.0 * torch.log(1.0 + f0 / 700.0)
        mask = f0_mel > 0
        f0_mel[mask] = (f0_mel[mask] - self._f0_mel_min) * 254.0 / (
            self._f0_mel_max - self._f0_mel_min
        ) + 1.0
        f0_mel[f0_mel <= 1.0] = 1.0
        f0_mel[f0_mel > 255.0] = 255.0
        f0_coarse = torch.round(f0_mel).long()
        return f0_coarse, f0
 
    def _get_f0_rmvpe(self, x_16k: torch.Tensor, f0_up_key: float) -> tuple[torch.Tensor, torch.Tensor]:
        if self._rmvpe is None:
            from rmvpe import RMVPE
            
            rmvpe_path = self._rmvpe_path
            if not rmvpe_path:
                files_dir = Path(__file__).parent / "files"
                alt_rmvpe = files_dir / "rmvpe.pt"
                if alt_rmvpe.exists():
                    rmvpe_path = str(alt_rmvpe)
            
            if not os.path.exists(rmvpe_path):
                 # Try finding in files if only filename given
                 if self._rmvpe_path and not os.path.isabs(self._rmvpe_path):
                     files_dir = Path(__file__).parent / "files"
                     alt = files_dir / self._rmvpe_path
                     if alt.exists():
                         rmvpe_path = str(alt)

            if not os.path.exists(rmvpe_path):
                 raise FileNotFoundError(f"找不到 RMVPE 权重: {rmvpe_path}")

            self._rmvpe = RMVPE(
                rmvpe_path,
                is_half=self.is_half,
                device=self.device,
                use_jit=False,
            )
        f0 = self._rmvpe.infer_from_audio(x_16k, thred=0.03)
        f0 = f0 * pow(2.0, float(f0_up_key) / 12.0)
        return self._get_f0_post(f0)
 
    def _get_f0_harvest(self, x_16k: torch.Tensor, f0_up_key: float) -> tuple[torch.Tensor, torch.Tensor]:
        import pyworld

        x = x_16k.detach().cpu().numpy().astype(np.float64)
        f0, t = pyworld.harvest(
            x,
            fs=16000,
            f0_ceil=float(self._f0_max),
            f0_floor=65.0,
            frame_period=10.0,
        )
        f0 = pyworld.stonemask(x, f0, t, 16000)
        f0 = f0.astype(np.float32) * pow(2.0, float(f0_up_key) / 12.0)
        return self._get_f0_post(f0)
 
    def _get_f0_pm(self, x_16k: torch.Tensor, f0_up_key: float) -> tuple[torch.Tensor, torch.Tensor]:
        import parselmouth

        x = x_16k.detach().cpu().numpy().astype(np.float32)
        p_len = x.shape[0] // 160 + 1
        f0_min = 65.0
        l_pad = int(np.ceil(1.5 / f0_min * 16000))
        r_pad = l_pad + 1
        s = parselmouth.Sound(np.pad(x, (l_pad, r_pad)), 16000).to_pitch_ac(
            time_step=0.01,
            voicing_threshold=0.6,
            pitch_floor=f0_min,
            pitch_ceiling=float(self._f0_max),
        )
        f0 = s.selected_array["frequency"].astype(np.float32)
        if f0.shape[0] < p_len:
            pad = np.zeros(p_len - f0.shape[0], dtype=np.float32)
            f0 = np.concatenate([f0, pad], axis=0)
        f0 = f0[:p_len] * pow(2.0, float(f0_up_key) / 12.0)
        return self._get_f0_post(f0)
 
    def get_f0(
        self,
        x_16k: torch.Tensor,
        *,
        f0_up_key: float,
        method: str,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        method = str(method or "rmvpe").lower()
        if method == "rmvpe":
            return self._get_f0_rmvpe(x_16k, f0_up_key)
        if method == "harvest":
            return self._get_f0_harvest(x_16k, f0_up_key)
        if method == "pm":
            return self._get_f0_pm(x_16k, f0_up_key)
        raise RuntimeError(f"不支持的 f0method: {method}")
 
    def infer(
        self,
        input_wav_16k: torch.Tensor,
        *,
        block_frame_16k: int,
        skip_head: int,
        return_length: int,
        f0method: str,
    ) -> torch.Tensor:
        self._ensure_models_loaded()
        assert self._hubert is not None
        assert self._net_g is not None
        assert self._info is not None
 
        input_wav_16k = input_wav_16k.to(self.device, dtype=torch.float16 if self.is_half else torch.float32)
        with torch.no_grad():
            feats_in = input_wav_16k.view(1, -1)
            padding_mask = torch.zeros_like(feats_in, dtype=torch.bool, device=self.device)
            output_layer = 9 if self._info.version == "v1" else 12
            logits = self._hubert.extract_features(
                source=feats_in,
                padding_mask=padding_mask,
                output_layer=output_layer,
            )
            feats = self._hubert.final_proj(logits[0]) if self._info.version == "v1" else logits[0]
            feats = torch.cat((feats, feats[:, -1:, :]), 1)
 
        if self._faiss_index is not None and self._faiss_big_npy is not None and self._index_rate > 0.0:
            try:
                npy = feats[0][skip_head // 2 :].detach().cpu().numpy().astype("float32")
                score, ix = self._faiss_index.search(npy, k=8)
                if (ix >= 0).all():
                    if torch.is_tensor(self._faiss_big_npy):
                        score = torch.from_numpy(score).to(self.device)
                        ix = torch.from_numpy(ix).to(self.device)
                        
                        weight = torch.square(1.0 / score)
                        weight /= weight.sum(dim=1, keepdim=True)
                        
                        npy2 = self._faiss_big_npy[ix.long()]
                        if self.is_half:
                            npy2 = npy2.half()
                        
                        weight = weight.unsqueeze(2)
                        npy2 = torch.sum(npy2 * weight, dim=1)
                        feats_mix = npy2.unsqueeze(0)
                    else:
                        weight = np.square(1.0 / score)
                        weight /= weight.sum(axis=1, keepdims=True)
                        npy2 = np.sum(self._faiss_big_npy[ix] * np.expand_dims(weight, axis=2), axis=1)
                        if self.is_half:
                            npy2 = npy2.astype("float16")
                        feats_mix = torch.from_numpy(npy2).unsqueeze(0).to(self.device)

                    feats[0][skip_head // 2 :] = feats_mix * float(self._index_rate) + (1.0 - float(self._index_rate)) * feats[
                        0
                    ][skip_head // 2 :]
            except Exception as e:
                print(f"Faiss error: {e}")
                pass
 
        p_len_int = int(input_wav_16k.shape[0] // 160)
        factor = pow(2.0, float(self.formant_shift) / 12.0)
        return_length2_int = int(np.ceil(float(return_length) * factor))
 
        if int(self._info.if_f0) == 1:
            f0_extractor_frame = int(block_frame_16k) + 800
            if str(f0method).lower() == "rmvpe":
                f0_extractor_frame = 5120 * ((f0_extractor_frame - 1) // 5120 + 1) - 160
            seg = input_wav_16k[-f0_extractor_frame:]
            pitch, pitchf = self.get_f0(
                seg,
                f0_up_key=float(self.f0_up_key) - float(self.formant_shift),
                method=f0method,
            )
            shift = int(block_frame_16k) // 160
            self._cache_pitch[:-shift] = self._cache_pitch[shift:].clone()
            self._cache_pitchf[:-shift] = self._cache_pitchf[shift:].clone()
            self._cache_pitch[4 - pitch.shape[0] :] = pitch[3:-1]
            self._cache_pitchf[4 - pitch.shape[0] :] = pitchf[3:-1]
            cache_pitch = self._cache_pitch[None, -p_len_int:]
            cache_pitchf = self._cache_pitchf[None, -p_len_int:] * (float(return_length2_int) / float(return_length))
 
        feats = F.interpolate(feats.permute(0, 2, 1), scale_factor=2.0, mode="nearest").permute(0, 2, 1)
        feats = feats[:, :p_len_int, :]
        p_len = torch.LongTensor([p_len_int]).to(self.device)
        sid = torch.LongTensor([0]).to(self.device)
        skip_head_t = torch.LongTensor([int(skip_head)]).to(self.device)
        return_length_t = torch.LongTensor([int(return_length)]).to(self.device)
        return_length2_t = torch.LongTensor([int(return_length2_int)]).to(self.device)
 
        with torch.no_grad():
            if int(self._info.if_f0) == 1:
                infered_audio, _, _ = self._net_g.infer(
                    feats,
                    p_len,
                    cache_pitch,
                    cache_pitchf,
                    sid,
                    skip_head_t,
                    return_length_t,
                    return_length2_t,
                )
            else:
                infered_audio, _, _ = self._net_g.infer(
                    feats,
                    p_len,
                    sid,
                    skip_head_t,
                    return_length_t,
                    return_length2_t,
                )
 
        infered_audio = infered_audio.squeeze(1).float()
        base_units = int(self._info.tgt_sr // 100)
        upp_units = int(np.floor(factor * float(base_units)))
        if upp_units <= 0:
            upp_units = base_units
        if upp_units != base_units:
            need = int(return_length) * upp_units
            if infered_audio.shape[1] >= need:
                infered_audio = infered_audio[:, :need]
            infered_audio = _resample_units(infered_audio, upp_units, base_units)
 
        return infered_audio.squeeze()
