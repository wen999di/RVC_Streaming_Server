import os
import logging
import time
from dataclasses import dataclass
from pathlib import Path
from collections import OrderedDict
from typing import Optional
import threading
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
 
 
def _file_identity(path: str) -> tuple[str, int, int]:
    resolved = Path(path).resolve()
    st = resolved.stat()
    return str(resolved), int(st.st_mtime_ns), int(st.st_size)


_HUBERT_CACHE: "OrderedDict[tuple, torch.nn.Module]" = OrderedDict()
_RMVPE_CACHE: "OrderedDict[tuple, object]" = OrderedDict()
_FCPE_CACHE: dict[str, object] = {}
_SHARED_NET_G_CACHE: "OrderedDict[tuple, dict]" = OrderedDict()
_SHARED_NET_G_LOCK = threading.RLock()
_SHARED_NET_G_MAX = max(1, int(os.environ.get("RVC_MODEL_CACHE_SIZE", "4")))
_DEVICE_INFER_LOCKS: dict[str, threading.Lock] = {}
_DEVICE_INFER_LOCKS_GUARD = threading.Lock()
_COMPONENT_CACHE_LOCK = threading.RLock()
_BASE_MODEL_CACHE_MAX = max(1, int(os.environ.get("RVC_BASE_MODEL_CACHE_SIZE", "2")))
_INDEX_CACHE_MAX = max(1, int(os.environ.get("RVC_INDEX_CACHE_SIZE", "4")))
_FAISS_CACHE: "OrderedDict[tuple, tuple[object, object, object | None]]" = OrderedDict()


def _device_infer_lock(device: torch.device) -> threading.Lock:
    key = str(device)
    with _DEVICE_INFER_LOCKS_GUARD:
        lock = _DEVICE_INFER_LOCKS.get(key)
        if lock is None:
            lock = threading.Lock()
            _DEVICE_INFER_LOCKS[key] = lock
        return lock


def _load_rmvpe(device: torch.device, is_half: bool, rmvpe_path: str):
    key = (str(device), bool(is_half), *_file_identity(rmvpe_path))
    with _COMPONENT_CACHE_LOCK:
        cached = _RMVPE_CACHE.get(key)
        if cached is not None:
            _RMVPE_CACHE.move_to_end(key)
            return cached
        from rmvpe import RMVPE
        instance = RMVPE(rmvpe_path, is_half=is_half, device=device, use_jit=False)
        _RMVPE_CACHE[key] = instance
        _RMVPE_CACHE.move_to_end(key)
        while len(_RMVPE_CACHE) > _BASE_MODEL_CACHE_MAX:
            _RMVPE_CACHE.popitem(last=False)
        return instance


def _load_fcpe(device: torch.device):
    key = str(device)
    with _COMPONENT_CACHE_LOCK:
        cached = _FCPE_CACHE.get(key)
        if cached is not None:
            return cached
        from torchfcpe import spawn_bundled_infer_model
        model = spawn_bundled_infer_model(device)
        _FCPE_CACHE[key] = model
        return model
 

def _load_hubert(device: torch.device, is_half: bool, hubert_path: str) -> torch.nn.Module:
    key = (str(device), bool(is_half), *_file_identity(hubert_path))
    with _COMPONENT_CACHE_LOCK:
        cached = _HUBERT_CACHE.get(key)
        if cached is not None:
            _HUBERT_CACHE.move_to_end(key)
            return cached
        from hubert import load_hubert
        hubert = load_hubert(hubert_path, device, is_half)
        _HUBERT_CACHE[key] = hubert
        _HUBERT_CACHE.move_to_end(key)
        while len(_HUBERT_CACHE) > _BASE_MODEL_CACHE_MAX:
            _HUBERT_CACHE.popitem(last=False)
        return hubert


def _load_faiss(device: torch.device, index_path: str):
    resolved, mtime_ns, file_size = _file_identity(index_path)
    key = (resolved, mtime_ns, file_size, str(device))
    with _COMPONENT_CACHE_LOCK:
        cached = _FAISS_CACHE.get(key)
        if cached is not None:
            _FAISS_CACHE.move_to_end(key)
            return cached

        import faiss
        index = faiss.read_index(resolved)
        big_npy = index.reconstruct_n(0, index.ntotal)
        resources = None
        if device.type == "cuda":
            resources = faiss.StandardGpuResources()
            gpu_id = 0 if device.index is None else int(device.index)
            index = faiss.index_cpu_to_gpu(resources, gpu_id, index)
            if isinstance(big_npy, np.ndarray):
                big_npy = torch.from_numpy(big_npy).to(device)
        _FAISS_CACHE[key] = (index, big_npy, resources)
        _FAISS_CACHE.move_to_end(key)
        while len(_FAISS_CACHE) > _INDEX_CACHE_MAX:
            _FAISS_CACHE.popitem(last=False)
        return index, big_npy, resources
 
 
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
        self._hubert_identity: tuple[str, int, int] | None = None
        self._rmvpe_identity: tuple[str, int, int] | None = None
        self._index_identity: tuple[str, int, int] | None = None

        self._hubert: Optional[torch.nn.Module] = None
        self._net_g: Optional[torch.nn.Module] = None
        self._info: Optional[LoadedModelInfo] = None
        self._acquired_model_key: tuple | None = None
        self._last_unloaded_model_path: str = ""
 
        self._faiss_index = None
        self._faiss_big_npy = None
        self._faiss_resource = None
 
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

    def _model_key(self, model_path: str) -> tuple:
        resolved, mtime_ns, file_size = _file_identity(model_path)
        return (resolved, mtime_ns, file_size, str(self.device), bool(self.is_half))

    def get_loaded_model_paths(self) -> list[str]:
        with _SHARED_NET_G_LOCK:
            return [
                key[0] for key in _SHARED_NET_G_CACHE.keys()
                if key[3] == str(self.device) and key[4] == bool(self.is_half)
            ]

    def reset_stream_state(self) -> None:
        self._cache_pitch.zero_()
        self._cache_pitchf.zero_()

    def _release_active_model(self) -> None:
        key = self._acquired_model_key
        if key is not None:
            with _SHARED_NET_G_LOCK:
                entry = _SHARED_NET_G_CACHE.get(key)
                if entry is not None:
                    entry["users"] = max(0, int(entry.get("users", 0)) - 1)
                    _SHARED_NET_G_CACHE.move_to_end(key)
        self._acquired_model_key = None
        self._net_g = None
        self._info = None

    def close(self) -> None:
        self.reset_stream_state()
        self._release_active_model()

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
    ) -> bool:
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

        model_identity_changed = False
        if model_path and self._acquired_model_key is not None:
            try:
                model_identity_changed = self._model_key(model_path) != self._acquired_model_key
            except FileNotFoundError:
                model_identity_changed = True

        model_asset_changed = model_path != self._model_path or model_identity_changed
        if model_asset_changed:
            self._release_active_model()
            self._model_path = model_path
            self.reset_stream_state()

        hubert_identity = _file_identity(hubert_path) if hubert_path else None
        hubert_asset_changed = hubert_path != self._hubert_path or hubert_identity != self._hubert_identity
        if hubert_asset_changed:
            self._hubert_path = hubert_path
            self._hubert_identity = hubert_identity
            self._hubert = None  # Reload required

        rmvpe_identity = _file_identity(rmvpe_path) if rmvpe_path else None
        rmvpe_asset_changed = rmvpe_path != self._rmvpe_path or rmvpe_identity != self._rmvpe_identity
        if rmvpe_asset_changed:
            self._rmvpe_path = rmvpe_path
            self._rmvpe_identity = rmvpe_identity
            self._rmvpe = None  # Reload required

        index_identity = _file_identity(index_path) if index_path else None
        index_asset_changed = index_path != self._index_path or index_identity != self._index_identity
        if index_asset_changed:
            self._index_path = index_path
            self._index_identity = index_identity
            self._faiss_index = None
            self._faiss_big_npy = None
            self._faiss_resource = None
        self._index_rate = index_rate

        return bool(model_asset_changed or hubert_asset_changed or rmvpe_asset_changed or index_asset_changed)

    def prepare(self, f0method: str = "rmvpe") -> LoadedModelInfo:
        with _device_infer_lock(self.device):
            return self._prepare_locked(f0method)

    def _prepare_locked(self, f0method: str) -> LoadedModelInfo:
        if not self._model_path:
            raise RuntimeError("缺少 model_path")

        started = time.perf_counter()

        stage_started = time.perf_counter()
        self._ensure_hubert_loaded()
        hubert_ms = (time.perf_counter() - stage_started) * 1000.0

        stage_started = time.perf_counter()
        self._ensure_active_model_loaded()
        voice_ms = (time.perf_counter() - stage_started) * 1000.0

        stage_started = time.perf_counter()
        self._ensure_index_loaded()
        index_ms = (time.perf_counter() - stage_started) * 1000.0

        stage_started = time.perf_counter()
        self._ensure_f0_model_loaded(f0method)
        f0_ms = (time.perf_counter() - stage_started) * 1000.0

        assert self._info is not None
        logging.info(
            "Model prepare stages: hubert=%.1fms voice=%.1fms index=%.1fms f0=%.1fms total=%.1fms device=%s",
            hubert_ms,
            voice_ms,
            index_ms,
            f0_ms,
            (time.perf_counter() - started) * 1000.0,
            self.device,
        )
        return self._info

    def warmup(self, f0method: str = "rmvpe") -> LoadedModelInfo:
        self.prepare(f0method)
        # Perform model loading and dummy inference under the same per-device
        # lock used by live inference, preventing preload/inference VRAM races.
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

        assert self._info is not None
        return self._info
 
    def _ensure_models_loaded(self) -> None:
        if not self._model_path:
            raise RuntimeError("缺少 model_path")

        self._ensure_hubert_loaded()
        self._ensure_active_model_loaded()
        self._ensure_index_loaded()

    def _ensure_index_loaded(self) -> None:
        if self._index_rate > 0.0 and self._index_path:
            if self._faiss_index is None:
                if not os.path.exists(self._index_path):
                    raise FileNotFoundError(f"找不到 index：{self._index_path}")
                self._faiss_index, self._faiss_big_npy, self._faiss_resource = _load_faiss(
                    self.device, self._index_path
                )

    def _ensure_f0_model_loaded(self, f0method: str) -> None:
        if self._info is not None and int(self._info.if_f0) != 1:
            return

        method = str(f0method or "rmvpe").lower()
        if method == "rmvpe":
            self._ensure_rmvpe_loaded()
            return
        if method == "fcpe":
            _load_fcpe(self.device)
            return
        raise RuntimeError(f"不支持的 f0method: {method}")

    def preload_model(self, model_path: str) -> dict:
        with _device_infer_lock(self.device):
            return self._preload_model_locked(model_path)

    def _preload_model_locked(self, model_path: str) -> dict:
        model_path = str(model_path or "")
        if not model_path:
            raise RuntimeError("缺少 model_path")
        self._ensure_hubert_loaded()
        key = self._model_key(model_path)
        evicted_paths: list[str] = []
        with _SHARED_NET_G_LOCK:
            if key in _SHARED_NET_G_CACHE:
                _SHARED_NET_G_CACHE.move_to_end(key)
                return {"loaded_paths": self.get_loaded_model_paths(), "evicted_paths": evicted_paths}

            while True:
                try:
                    net_g, info = self._load_net_g_from_path(model_path)
                    _SHARED_NET_G_CACHE[key] = {"net": net_g, "info": info, "users": 0}
                    _SHARED_NET_G_CACHE.move_to_end(key)
                    break
                except RuntimeError as e:
                    if self.device.type == "cuda" and "out of memory" in str(e).lower():
                        evicted = self._evict_one_cached_model()
                        if not evicted:
                            raise
                        evicted_paths.append(evicted)
                        continue
                    raise

            while len(_SHARED_NET_G_CACHE) > _SHARED_NET_G_MAX:
                evicted = self._evict_one_cached_model(keep_key=key)
                if not evicted:
                    break
                evicted_paths.append(evicted)

        return {"loaded_paths": self.get_loaded_model_paths(), "evicted_paths": evicted_paths}

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
        if not self._model_path:
            raise RuntimeError("缺少 model_path")
        if self._acquired_model_key is not None and self._net_g is not None:
            return
        desired_key = self._model_key(self._model_path)
        self._release_active_model()

        with _SHARED_NET_G_LOCK:
            entry = _SHARED_NET_G_CACHE.get(desired_key)
            if entry is None:
                while True:
                    try:
                        net_g, info = self._load_net_g_from_path(self._model_path)
                        entry = {"net": net_g, "info": info, "users": 0}
                        _SHARED_NET_G_CACHE[desired_key] = entry
                        break
                    except RuntimeError as e:
                        if self.device.type == "cuda" and "out of memory" in str(e).lower():
                            evicted = self._evict_one_cached_model(keep_key=desired_key)
                            if not evicted:
                                raise
                            continue
                        raise
            entry["users"] = int(entry.get("users", 0)) + 1
            _SHARED_NET_G_CACHE.move_to_end(desired_key)
            self._acquired_model_key = desired_key
            self._net_g = entry["net"]
            self._info = entry["info"]

            while len(_SHARED_NET_G_CACHE) > _SHARED_NET_G_MAX:
                if not self._evict_one_cached_model(keep_key=desired_key):
                    break

    def _evict_one_cached_model(self, keep_key: tuple | None = None) -> str:
        with _SHARED_NET_G_LOCK:
            victim_key = None
            for key, entry in _SHARED_NET_G_CACHE.items():
                if key == keep_key or int(entry.get("users", 0)) > 0:
                    continue
                victim_key = key
                break
            if victim_key is None:
                return ""

            entry = _SHARED_NET_G_CACHE.pop(victim_key)
            victim_net = entry.get("net")
            try:
                if victim_net is not None:
                    victim_net.to("cpu")
            except Exception:
                pass
            del victim_net
            if self.device.type == "cuda":
                try:
                    torch.cuda.empty_cache()
                except Exception:
                    pass
            self._last_unloaded_model_path = victim_key[0]
            return victim_key[0]

    def _load_net_g_from_path(self, model_path: str) -> tuple[torch.nn.Module, LoadedModelInfo]:
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"找不到音色模型：{model_path}")

        from models import (
            SynthesizerTrnMs256NSFsid,
            SynthesizerTrnMs256NSFsid_nono,
            SynthesizerTrnMs768NSFsid,
            SynthesizerTrnMs768NSFsid_nono,
        )

        # RVC checkpoints contain plain tensors/config data; keep unpickling in
        # weights-only mode and stage on CPU to avoid an avoidable GPU memory spike.
        cpt = torch.load(model_path, map_location="cpu", weights_only=True)

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
        net_g.load_state_dict(cpt["weight"], strict=False)
        del cpt
        net_g = net_g.to(self.device)
        net_g = net_g.half() if self.is_half else net_g.float()
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
        self._ensure_rmvpe_loaded()
        assert self._rmvpe is not None
        f0 = self._rmvpe.infer_from_audio(x_16k, thred=0.03)
        f0 = f0 * pow(2.0, float(f0_up_key) / 12.0)
        return self._get_f0_post(f0)

    def _ensure_rmvpe_loaded(self) -> None:
        if self._rmvpe is not None:
            return

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

    def _get_f0_fcpe(self, x_16k: torch.Tensor, f0_up_key: float) -> tuple[torch.Tensor, torch.Tensor]:
        model = _load_fcpe(self.device)
        x = x_16k.unsqueeze(0).float().to(self.device)
        f0 = model.infer(x, sr=16000, decoder_mode="local_argmax", threshold=0.006)
        f0 = f0.squeeze().float()
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
        with _device_infer_lock(self.device):
            return self._infer_locked(
                input_wav_16k,
                block_frame_16k=block_frame_16k,
                skip_head=skip_head,
                return_length=return_length,
                f0method=f0method,
            )

    @torch.inference_mode()
    def _infer_locked(
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
                query = feats[0][skip_head // 2 :].float().contiguous()
                score = ix = None
                if self.device.type == "cuda":
                    try:
                        import faiss.contrib.torch_utils  # noqa: F401
                        score, ix = self._faiss_index.search(query, k=8)
                    except Exception:
                        score = ix = None
                if score is None or ix is None:
                    score, ix = self._faiss_index.search(query.detach().cpu().numpy(), k=8)

                if torch.is_tensor(ix):
                    valid = bool(torch.all(ix >= 0).item())
                else:
                    valid = bool((ix >= 0).all())

                if valid:
                    if torch.is_tensor(self._faiss_big_npy):
                        if not torch.is_tensor(score):
                            score = torch.from_numpy(score).to(self.device)
                        else:
                            score = score.to(self.device)
                        if not torch.is_tensor(ix):
                            ix = torch.from_numpy(ix).to(self.device)
                        else:
                            ix = ix.to(self.device)
                        score = score.clamp_min(1e-6)
                        weight = torch.reciprocal(score).square()
                        weight /= weight.sum(dim=1, keepdim=True).clamp_min(1e-12)
                        npy2 = self._faiss_big_npy[ix.long()]
                        if self.is_half:
                            npy2 = npy2.half()
                        npy2 = torch.sum(npy2 * weight.unsqueeze(2).to(npy2.dtype), dim=1)
                        feats_mix = npy2.unsqueeze(0)
                    else:
                        score = np.maximum(np.asarray(score), 1e-6)
                        weight = np.square(1.0 / score)
                        weight /= np.maximum(weight.sum(axis=1, keepdims=True), 1e-12)
                        npy2 = np.sum(self._faiss_big_npy[ix] * np.expand_dims(weight, axis=2), axis=1)
                        if self.is_half:
                            npy2 = npy2.astype("float16")
                        feats_mix = torch.from_numpy(npy2).unsqueeze(0).to(self.device)

                    feats[0][skip_head // 2 :] = (
                        feats_mix * float(self._index_rate)
                        + (1.0 - float(self._index_rate)) * feats[0][skip_head // 2 :]
                    )
            except Exception as e:
                print(f"Faiss error: {e}")
 
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
