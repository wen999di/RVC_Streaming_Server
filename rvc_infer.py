import os
from dataclasses import dataclass
from pathlib import Path
from collections import OrderedDict
from typing import Optional
import numpy as np
import torch
import torch.nn.functional as F


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
_RMVPE_CACHE: dict[tuple[str, bool, str], object] = {}
_FCPE_CACHE: dict[str, object] = {}


def _load_rmvpe(device: torch.device, is_half: bool, rmvpe_path: str):
    key = (str(device), bool(is_half), str(rmvpe_path))
    cached = _RMVPE_CACHE.get(key)
    if cached is not None:
        return cached
    from rmvpe import RMVPE
    instance = RMVPE(rmvpe_path, is_half=is_half, device=device, use_jit=False)
    _RMVPE_CACHE[key] = instance
    return instance


def _load_fcpe(device: torch.device):
    key = str(device)
    cached = _FCPE_CACHE.get(key)
    if cached is not None:
        return cached
    from torchfcpe import spawn_bundled_infer_model
    model = spawn_bundled_infer_model(device)
    _FCPE_CACHE[key] = model
    return model
 
 
def _load_hubert(device: torch.device, is_half: bool, hubert_path: str) -> torch.nn.Module:
    key = (str(device), bool(is_half), str(hubert_path))
    cached = _HUBERT_CACHE.get(key)
    if cached is not None:
        return cached
 
    from hubert import load_hubert
    hubert = load_hubert(hubert_path, device, is_half)
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
        self._net_g_cache: "OrderedDict[str, tuple[torch.nn.Module, LoadedModelInfo]]" = OrderedDict()
        self._last_unloaded_model_path: str = ""
 
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

    @property
    def last_unloaded_model_path(self) -> str:
        return self._last_unloaded_model_path

    def get_loaded_model_paths(self) -> list[str]:
        return list(self._net_g_cache.keys())
 
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

        self._ensure_hubert_loaded()
        self._ensure_active_model_loaded()

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

    def preload_model(self, model_path: str) -> dict:
        model_path = str(model_path or "")
        if not model_path:
            raise RuntimeError("缺少 model_path")

        self._ensure_hubert_loaded()
        evicted_paths: list[str] = []

        if model_path in self._net_g_cache:
            self._activate_cached_model(model_path)
            return {
                "loaded_paths": self.get_loaded_model_paths(),
                "evicted_paths": evicted_paths,
            }

        while True:
            try:
                net_g, info = self._load_net_g_from_path(model_path)
                self._net_g_cache[model_path] = (net_g, info)
                self._net_g_cache.move_to_end(model_path)
                break
            except RuntimeError as e:
                if self.device.type == "cuda" and "out of memory" in str(e).lower():
                    evicted = self._evict_one_cached_model(keep_path=self._model_path)
                    if not evicted:
                        raise
                    evicted_paths.append(evicted)
                    continue
                raise

        if model_path == self._model_path:
            self._activate_cached_model(model_path)

        return {
            "loaded_paths": self.get_loaded_model_paths(),
            "evicted_paths": evicted_paths,
        }

    def _ensure_hubert_loaded(self) -> None:
        if self._hubert is not None:
            return

        hubert_path = self._hubert_path
        if not hubert_path:
            files_dir = Path(__file__).parent / "files"
            alt_hubert = files_dir / "hubert_base.pt"
            if alt_hubert.exists():
                hubert_path = str(alt_hubert)

        if not os.path.exists(hubert_path):
            if self._hubert_path and not os.path.isabs(self._hubert_path):
                files_dir = Path(__file__).parent / "files"
                alt = files_dir / self._hubert_path
                if alt.exists():
                    hubert_path = str(alt)

        if not os.path.exists(hubert_path):
            raise FileNotFoundError(f"找不到 HuBERT 权重：{hubert_path}")

        self._hubert = _load_hubert(self.device, self.is_half, hubert_path)

    def _ensure_active_model_loaded(self) -> None:
        if self._model_path in self._net_g_cache:
            self._activate_cached_model(self._model_path)
            return

        while True:
            try:
                net_g, info = self._load_net_g_from_path(self._model_path)
                self._net_g_cache[self._model_path] = (net_g, info)
                self._net_g_cache.move_to_end(self._model_path)
                self._activate_cached_model(self._model_path)
                return
            except RuntimeError as e:
                if self.device.type == "cuda" and "out of memory" in str(e).lower():
                    evicted = self._evict_one_cached_model(keep_path=self._model_path)
                    if not evicted:
                        raise
                    continue
                raise

    def _activate_cached_model(self, model_path: str) -> None:
        net_g, info = self._net_g_cache[model_path]
        self._net_g = net_g
        self._info = info
        self._net_g_cache.move_to_end(model_path)

    def _evict_one_cached_model(self, keep_path: str = "") -> str:
        if not self._net_g_cache:
            return ""

        victim_path = ""
        for path in self._net_g_cache.keys():
            if keep_path and path == keep_path:
                continue
            victim_path = path
            break

        if not victim_path:
            if keep_path in self._net_g_cache and len(self._net_g_cache) == 1:
                return ""
            victim_path = next(iter(self._net_g_cache.keys()))

        victim_net, _ = self._net_g_cache.pop(victim_path)
        try:
            victim_net.to("cpu")
        except Exception:
            pass
        del victim_net

        if self._net_g is not None and victim_path == self._model_path:
            self._net_g = None
            self._info = None

        if self.device.type == "cuda":
            try:
                torch.cuda.empty_cache()
            except Exception:
                pass

        self._last_unloaded_model_path = victim_path
        return victim_path

    def _load_net_g_from_path(self, model_path: str) -> tuple[torch.nn.Module, LoadedModelInfo]:
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"找不到音色模型：{model_path}")

        from models import (
            SynthesizerTrnMs256NSFsid,
            SynthesizerTrnMs256NSFsid_nono,
            SynthesizerTrnMs768NSFsid,
            SynthesizerTrnMs768NSFsid_nono,
        )

        load_map_location = self.device if self.device.type == "cuda" else "cpu"
        try:
            try:
                cpt = torch.load(model_path, map_location=load_map_location, weights_only=False)
            except TypeError:
                cpt = torch.load(model_path, map_location=load_map_location)
        except RuntimeError as e:
            if self.device.type == "cuda" and "out of memory" in str(e).lower():
                try:
                    cpt = torch.load(model_path, map_location="cpu", weights_only=False)
                except TypeError:
                    cpt = torch.load(model_path, map_location="cpu")
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
        else:
            raise RuntimeError(f"未知模型版本: {version}")

        del net_g.enc_q
        net_g = net_g.to(self.device)
        net_g = net_g.half() if self.is_half else net_g.float()
        net_g.load_state_dict(cpt["weight"], strict=False)
        net_g.eval()

        info = LoadedModelInfo(tgt_sr=tgt_sr, if_f0=if_f0, version=version)
        return net_g, info

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

            self._rmvpe = _load_rmvpe(self.device, self.is_half, rmvpe_path)
        f0 = self._rmvpe.infer_from_audio(x_16k, thred=0.03)
        f0 = f0 * pow(2.0, float(f0_up_key) / 12.0)
        return self._get_f0_post(f0)
 
    def _get_f0_fcpe(self, x_16k: torch.Tensor, f0_up_key: float) -> tuple[torch.Tensor, torch.Tensor]:
        model = _load_fcpe(self.device)
        x = x_16k.unsqueeze(0).float().to(self.device)
        f0 = model.infer(x, sr=16000, decoder_mode="local_argmax", threshold=0.006)
        f0 = f0.squeeze().cpu().numpy().astype(np.float32)
        f0 = f0 * pow(2.0, float(f0_up_key) / 12.0)
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
        if method == "fcpe":
            return self._get_f0_fcpe(x_16k, f0_up_key)
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
