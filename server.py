import asyncio
import contextlib
import websockets
import logging
import hashlib
import hmac
from logging.handlers import RotatingFileHandler
import time
import threading
import ssl
import json
import os
import glob
from pathlib import Path
from collections.abc import Mapping
from rvc_core import RVCCore
from audio_protocol import (
    AudioInputFrame, FLAG_DISCONTINUITY, build_audio_output_frame, parse_audio_input_frame,
)
from file_transfer import UploadManager, parse_file_chunk_frame
from model_registry import ModelRegistry

# 全局状态
log_subscribers = set()
_ws_send_locks = {}
log_queue = asyncio.Queue(maxsize=1000)
_main_loop = None
upload_manager = UploadManager()
model_registry = ModelRegistry()

_pth_meta_cache: dict[str, tuple[float, int, dict | None]] = {}


def _safe_int(v, default=None):
    try:
        return int(v)
    except Exception:
        return default


def _compute_config_hash(config: dict) -> str:
    """
    计算配置 hash，用于客户端/服务端一致性校验。
    仅包含由客户端控制的键。
    浮点数统一保留 4 位小数以保证跨语言一致性。
    路径仅参与 basename，避免平台差异。
    """
    keys_to_hash = [
        "model_path", "index_path", "f0_up_key", "block_time",
        "crossfade_length", "extra_time",
        "formant_shift", "f0method", "index_rate", "passthrough",
        "silence_db_threshold", "silence_gate_atten",
        "input_noise_reduce", "output_noise_reduce", "noise_reduce_prop_decrease",
        "rms_mix_rate"
    ]

    float_keys = {
        "block_time", "crossfade_length", "extra_time",
        "formant_shift", "index_rate",
        "silence_db_threshold", "silence_gate_atten",
        "noise_reduce_prop_decrease",
        "rms_mix_rate"
    }

    parts = []
    for k in sorted(keys_to_hash):
        val = config.get(k)
        if k in ("model_path", "index_path"):
            # 仅使用 basename，避免路径差异导致 hash 不一致。
            s_val = os.path.basename(str(val)) if val else ""
            parts.append(f"{k}={s_val}")
        elif k in float_keys:
            # 即使传入为 int，也按 float 格式化（例如 2 -> 2.0000）。
            try:
                f_val = float(val) if val is not None else 0.0
                s_val = f"{f_val:.4f}"
            except (ValueError, TypeError):
                s_val = "0.0000"
            parts.append(f"{k}={s_val}")
        elif isinstance(val, float):
            # 兼容未列入 float_keys 的其他 float 键（如存在）。
            s_val = f"{val:.4f}"
            parts.append(f"{k}={s_val}")
        else:
            parts.append(f"{k}={val}")

    # 构造确定性的字符串表示。
    raw_str = "|".join(parts)
    return hashlib.md5(raw_str.encode('utf-8')).hexdigest()


def _resolve_runtime_config(public_config: dict) -> dict:
    runtime = dict(public_config or {})
    files_dir = upload_manager.files_dir
    for k in ("model_path", "index_path", "hubert_path", "rmvpe_path"):
        v = runtime.get(k)
        if not v:
            runtime[k] = ""
            continue
        base = os.path.basename(str(v))
        candidate = files_dir / base
        if candidate.exists() and candidate.is_file():
            runtime[k] = str(candidate)
        else:
            logging.warning(f"Runtime path missing for {k}: {base}")
            runtime[k] = ""
    return runtime


def _try_parse_voice_model_pth_meta(path: Path) -> dict | None:
    try:
        import torch
    except Exception:
        return None
    try:
        cpt = torch.load(str(path), map_location="cpu", weights_only=True)
        if isinstance(cpt, Mapping) and "model" in cpt and isinstance(cpt.get("model"), Mapping):
            cpt = cpt["model"]
        if not isinstance(cpt, Mapping):
            return None
        version = cpt.get("version", None)
        sr = cpt.get("sr", None)
        f0 = cpt.get("f0", None)
        info = cpt.get("info", None)

        if not isinstance(version, str) or not version:
            return None
        if not isinstance(sr, str) or not sr:
            return None
        if not isinstance(f0, int):
            return None
        if not isinstance(info, str) or not info:
            return None

        return {
            "ok": True,
            "version": version,
            "sr": sr,
            "f0": f0,
            "info": info,
        }
    except Exception:
        return None


def _enrich_files_with_voice_meta(files: list[dict], files_dir: Path) -> list[dict]:
    for item in files:
        try:
            name = str(item.get("name") or "")
            if not name.lower().endswith(".pth"):
                continue
            mtime = float(item.get("mtime") or 0.0)
            size = _safe_int(item.get("size"), 0) or 0

            cached = _pth_meta_cache.get(name)
            if cached and cached[0] == mtime and cached[1] == size:
                meta = cached[2]
            else:
                meta = _try_parse_voice_model_pth_meta(files_dir / name)
                _pth_meta_cache[name] = (mtime, size, meta)

            if meta:
                item["voice_meta"] = meta
        except Exception:
            continue
    return files

# 确保 logs 目录存在。
LOG_DIR = "./logs"
os.makedirs(LOG_DIR, exist_ok=True)
# 当前日志文件名包含时间戳。
CURRENT_LOG_FILE = os.path.join(LOG_DIR, f"server_{time.strftime('%Y-%m-%d_%H-%M-%S')}.log")

def _enqueue_log_message(message: str) -> None:
    if log_queue.full():
        try:
            log_queue.get_nowait()
            log_queue.task_done()
        except asyncio.QueueEmpty:
            pass
    try:
        log_queue.put_nowait(message)
    except asyncio.QueueFull:
        pass


class WebSocketLogHandler(logging.Handler):
    """将日志写入 asyncio 队列，用于广播到 WebSocket 订阅者。"""
    def emit(self, record):
        try:
            msg = self.format(record)
            loop = _main_loop
            if loop is not None and not loop.is_closed():
                loop.call_soon_threadsafe(_enqueue_log_message, msg + "\n")
        except Exception:
            self.handleError(record)

# 配置日志
logging.basicConfig(level=logging.INFO, handlers=[])  # 清空默认 handler
root_logger = logging.getLogger()
root_logger.setLevel(logging.INFO)

# 1) 控制台输出
console_handler = logging.StreamHandler()
console_handler.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))
root_logger.addHandler(console_handler)

# 2) 文件输出（滚动）
file_handler = RotatingFileHandler(CURRENT_LOG_FILE, maxBytes=5*1024*1024, backupCount=5, encoding='utf-8')
file_handler.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))
root_logger.addHandler(file_handler)

# 3) WebSocket 广播
ws_handler = WebSocketLogHandler()
ws_handler.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))
root_logger.addHandler(ws_handler)

async def _ws_send(websocket, payload, *, timeout: float | None = None):
    lock = _ws_send_locks.get(websocket)
    if lock is None:
        lock = asyncio.Lock()
        _ws_send_locks[websocket] = lock
    async with lock:
        send_coro = websocket.send(payload)
        if timeout is None:
            return await send_coro
        return await asyncio.wait_for(send_coro, timeout=timeout)


async def log_broadcaster():
    """后台任务：将日志广播给订阅者。"""
    while True:
        msg = await log_queue.get()
        if log_subscribers:
            # 广播给所有订阅者
            for ws in list(log_subscribers):
                try:
                    await _ws_send(
                        ws,
                        json.dumps({"status": "ok", "type": "log_chunk", "content": msg}),
                        timeout=1.0,
                    )
                except Exception:
                    log_subscribers.discard(ws)
        log_queue.task_done()


class AudioProcessor:
    def __init__(self, config=None):
        self.config = config or {}
        self._lock = threading.RLock()
        self.core = RVCCore(_resolve_runtime_config(self.config))
        logging.info(f"AudioProcessor initialized with config: {self.config}")

    def update_config(self, config):
        with self._lock:
            self.config.update(config)
            changes = self.core.update_config(_resolve_runtime_config(self.config))
            logging.info(f"Config updated: {self.config}")
            return changes

    def reset_stream_state(self):
        with self._lock:
            self.core.reset_stream_state()

    def close(self):
        with self._lock:
            self.core.close()

    def warmup(self):
        with self._lock:
            return self.core.warmup()

    def preload_voice_model(self, model_path: str):
        with self._lock:
            return self.core.preload_voice_model(model_path)

    def process_packet(self, audio_data, ts_start_ns=None):
        with self._lock:
            block_bytes = max(4, int(self.core.block_frame) * int(self.core.bytes_per_sample))
            ns_per_sample = int(self.core.ns_per_sample)
            results = []
            for offset in range(0, len(audio_data), block_bytes):
                chunk = audio_data[offset : offset + block_bytes]
                chunk_ts = int(ts_start_ns or 0) + (offset // int(self.core.bytes_per_sample)) * ns_per_sample
                t0 = time.perf_counter()
                out_pcm, out_ts_ns = self.core.process_frame(chunk, chunk_ts)
                proc_ms = int(round((time.perf_counter() - t0) * 1000.0))
                if out_pcm:
                    results.append((out_pcm, out_ts_ns, proc_ms))
            return results


class RealtimeAudioSession:
    def __init__(self, websocket, processor: AudioProcessor):
        self.websocket = websocket
        self.processor = processor
        self.active_session_id = 0
        self.last_input_sequence = None
        self.output_sequence = 0
        self.input_queue = asyncio.Queue(maxsize=max(2, int(os.environ.get("RVC_AUDIO_INPUT_QUEUE", "8"))))
        self.output_queue = asyncio.Queue(maxsize=max(1, int(os.environ.get("RVC_AUDIO_OUTPUT_QUEUE", "1"))))
        self.max_input_backlog_samples = max(
            160,
            int(16000 * max(20, int(os.environ.get("RVC_MAX_INPUT_BACKLOG_MS", "200"))) / 1000),
        )
        self._queued_input_samples = 0
        self._state_lock = asyncio.Lock()
        self._worker_task = asyncio.create_task(self._worker_loop())
        self._sender_task = asyncio.create_task(self._sender_loop())
        self._pending_discontinuity = False
        self._output_discontinuity_pending = False

    @staticmethod
    def _drain(queue: asyncio.Queue) -> None:
        while True:
            try:
                queue.get_nowait()
                queue.task_done()
            except asyncio.QueueEmpty:
                return

    def _drain_input(self) -> None:
        self._drain(self.input_queue)
        self._queued_input_samples = 0

    async def reset(self, session_id: int | None = None) -> None:
        async with self._state_lock:
            if session_id is not None:
                self.active_session_id = int(session_id)
            self.last_input_sequence = None
            self.output_sequence = 0
            self._pending_discontinuity = True
            self._output_discontinuity_pending = True
            self._drain_input()
            self._drain(self.output_queue)
            await asyncio.to_thread(self.processor.reset_stream_state)

    async def apply_config(self, cfg: dict) -> dict:
        async with self._state_lock:
            changes = await asyncio.to_thread(self.processor.update_config, cfg)
            if changes.get("buffer_layout") or changes.get("model_runtime"):
                self.last_input_sequence = None
                self.output_sequence = 0
                self._pending_discontinuity = True
                self._output_discontinuity_pending = True
                self._drain_input()
                self._drain(self.output_queue)
                await asyncio.to_thread(self.processor.reset_stream_state)
            should_warmup = bool(changes.get("model_runtime"))
            if should_warmup and not self.processor.core.passthrough and self.processor.core.model_path:
                await asyncio.to_thread(self.processor.warmup)
            return changes

    async def enqueue(self, frame: AudioInputFrame) -> None:
        if not self.active_session_id or frame.session_id != self.active_session_id:
            return

        sample_count = len(frame.payload) // 4
        dropped = False
        if sample_count > self.max_input_backlog_samples:
            drop_samples = sample_count - self.max_input_backlog_samples
            frame = AudioInputFrame(
                session_id=frame.session_id,
                sequence=frame.sequence,
                sample_rate=frame.sample_rate,
                timestamp_ns=frame.timestamp_ns + drop_samples * (1_000_000_000 // 16000),
                flags=frame.flags | FLAG_DISCONTINUITY,
                payload=frame.payload[drop_samples * 4 :],
            )
            sample_count = self.max_input_backlog_samples
            dropped = True

        while self.input_queue.full() or self._queued_input_samples + sample_count > self.max_input_backlog_samples:
            try:
                old_frame, _ = self.input_queue.get_nowait()
                self.input_queue.task_done()
                self._queued_input_samples = max(0, self._queued_input_samples - len(old_frame.payload) // 4)
                dropped = True
            except asyncio.QueueEmpty:
                break

        if dropped:
            # The next worker iteration will reset inference state and then mark
            # its first generated output discontinuous.
            self._pending_discontinuity = True
        self.input_queue.put_nowait((frame, time.perf_counter()))
        self._queued_input_samples += sample_count

    async def _worker_loop(self) -> None:
        while True:
            frame, enqueue_time = await self.input_queue.get()
            self._queued_input_samples = max(0, self._queued_input_samples - len(frame.payload) // 4)
            try:
                if frame.session_id != self.active_session_id:
                    continue
                input_queue_ms = int(round((time.perf_counter() - enqueue_time) * 1000.0))
                async with self._state_lock:
                    if frame.session_id != self.active_session_id:
                        continue
                    discontinuity = self._pending_discontinuity or bool(frame.flags & FLAG_DISCONTINUITY)
                    expected = None if self.last_input_sequence is None else ((self.last_input_sequence + 1) & 0xFFFFFFFF)
                    if expected is not None and frame.sequence != expected:
                        discontinuity = True
                    if discontinuity:
                        await asyncio.to_thread(self.processor.reset_stream_state)
                        self._pending_discontinuity = False
                        self._output_discontinuity_pending = True
                    self.last_input_sequence = frame.sequence
                    outputs = await asyncio.to_thread(
                        self.processor.process_packet, frame.payload, frame.timestamp_ns
                    )
                    if frame.session_id != self.active_session_id:
                        continue

                for out_pcm, out_ts_ns, proc_ms in outputs:
                    dropped_output = False
                    while self.output_queue.full():
                        try:
                            self.output_queue.get_nowait()
                            self.output_queue.task_done()
                            dropped_output = True
                        except asyncio.QueueEmpty:
                            break

                    flags = FLAG_DISCONTINUITY if (self._output_discontinuity_pending or dropped_output) else 0
                    self._output_discontinuity_pending = False
                    item = (
                        frame.session_id, self.output_sequence, int(out_ts_ns or 0), proc_ms,
                        input_queue_ms, flags, out_pcm, time.perf_counter(),
                    )
                    self.output_sequence = (self.output_sequence + 1) & 0xFFFFFFFF
                    self.output_queue.put_nowait(item)
            finally:
                self.input_queue.task_done()

    async def _sender_loop(self) -> None:
        while True:
            item = await self.output_queue.get()
            try:
                session_id, sequence, ts_ns, proc_ms, input_queue_ms, flags, payload, enqueue_time = item
                if session_id != self.active_session_id:
                    continue
                output_queue_ms = int(round((time.perf_counter() - enqueue_time) * 1000.0))
                frame = build_audio_output_frame(
                    session_id=session_id, sequence=sequence, sample_rate=16000,
                    timestamp_ns=ts_ns, proc_ms=proc_ms, input_queue_ms=input_queue_ms,
                    output_queue_ms=output_queue_ms, flags=flags, payload=payload,
                )
                await _ws_send(self.websocket, frame)
            finally:
                self.output_queue.task_done()

    async def close(self) -> None:
        for task in (self._worker_task, self._sender_task):
            task.cancel()
        for task in (self._worker_task, self._sender_task):
            with contextlib.suppress(asyncio.CancelledError):
                await task
        self._drain_input()
        self._drain(self.output_queue)


async def binary_echo_handler(websocket):
    path = str(getattr(websocket, "path", "/") or "/")
    role = "audio" if path.rstrip("/").endswith("/audio") else "control"

    expected_token = os.environ.get("RVC_STREAMING_TOKEN", "")
    if expected_token:
        auth = str(websocket.request_headers.get("Authorization", ""))
        provided = auth[7:] if auth.startswith("Bearer ") else ""
        if not provided or not hmac.compare_digest(provided, expected_token):
            logging.warning(f"Rejected unauthorized {role} client: {websocket.remote_address}")
            await websocket.close(code=1008, reason="Unauthorized")
            return

    logging.info(f"Client connected: role={role} remote={websocket.remote_address}")
    processor = AudioProcessor()
    audio_session = RealtimeAudioSession(websocket, processor) if role == "audio" else None
    logging.info(f"Audio Processor initialized for role={role}")

    def _abs_voice_model_path(filename: str) -> str:
        base = os.path.basename(str(filename or ""))
        if not base:
            return ""
        return str((upload_manager.files_dir / base).resolve())

    def _attach_voice_runtime_state(voice: dict) -> dict:
        out = dict(voice or {})
        models = out.get("models") if isinstance(out.get("models"), list) else []

        inferer = getattr(getattr(processor, "core", None), "_inferer", None)
        loaded_paths = set()
        last_unloaded_path = ""
        try:
            if inferer is not None:
                loaded_paths = {str(Path(p).resolve()) for p in inferer.get_loaded_model_paths()}
                last_unloaded_path = str(getattr(inferer, "last_unloaded_model_path", "") or "")
                if last_unloaded_path:
                    last_unloaded_path = str(Path(last_unloaded_path).resolve())
        except Exception:
            loaded_paths = set()
            last_unloaded_path = ""

        out_models = []
        last_unloaded_id = ""
        for m in models:
            if not isinstance(m, dict):
                continue
            item = dict(m)
            full_path = _abs_voice_model_path(item.get("pth", ""))
            item["loaded"] = bool(full_path and full_path in loaded_paths)
            if full_path and last_unloaded_path and full_path == last_unloaded_path:
                last_unloaded_id = str(item.get("id") or "")
            out_models.append(item)

        out["models"] = out_models
        out["last_unloaded_id"] = last_unloaded_id
        return out

    async def _send_voice_models() -> None:
        voice = await asyncio.to_thread(model_registry.list_voice_models)
        voice = _attach_voice_runtime_state(voice)
        await _ws_send(websocket,
            json.dumps(
                {
                    "status": "ok",
                    "type": "voice_models",
                    "voice": voice,
                }
            )
        )

    try:
        async for message in websocket:
            # 根据消息类型进行处理
            if isinstance(message, str):
                # JSON 配置消息或命令
                try:
                    data = json.loads(message)

                    command = str(data.get("command") or "")
                    has_config = "config" in data
                    audio_commands = {"stream_start", "stream_stop"}
                    if role == "audio" and not (has_config or command in audio_commands):
                        await _ws_send(websocket, json.dumps({
                            "status": "error", "type": "endpoint_error",
                            "message": "command_not_allowed_on_audio_endpoint",
                        }))
                        continue
                    if role == "control" and (has_config or command in audio_commands):
                        await _ws_send(websocket, json.dumps({
                            "status": "error", "type": "endpoint_error",
                            "message": "audio_endpoint_required",
                        }))
                        continue

                    if command == "stream_start":
                        if audio_session is None:
                            await _ws_send(websocket, json.dumps({"status": "error", "type": "stream_error", "message": "audio_endpoint_required"}))
                            continue
                        session_id = int(data.get("session_id") or 0)
                        if session_id <= 0:
                            await _ws_send(websocket, json.dumps({"status": "error", "type": "stream_error", "message": "invalid_session_id"}))
                            continue
                        await audio_session.reset(session_id)
                        await _ws_send(websocket, json.dumps({"status": "ok", "type": "stream_started", "session_id": session_id, "protocol": 2}))

                    elif command == "stream_stop":
                        if audio_session is not None:
                            session_id = int(data.get("session_id") or 0)
                            if not session_id or session_id == audio_session.active_session_id:
                                await audio_session.reset(0)
                        await _ws_send(websocket, json.dumps({"status": "ok", "type": "stream_stopped"}))

                    # 1. 配置更新
                    elif "config" in data:
                        cfg = data["config"] if isinstance(data.get("config"), dict) else {}
                        seq = data.get("seq", None)
                        passthrough = bool(cfg.get("passthrough", False))

                        # 校验音色模型文件存在（路径解析统一由 _resolve_runtime_config 处理）
                        if "model_path" in cfg and not passthrough:
                            client_pth = os.path.basename(str(cfg.get("model_path") or ""))
                            if not client_pth or not (upload_manager.files_dir / client_pth).is_file():
                                logging.error(f"Config Error: Voice model not found. client_pth={client_pth}")
                                await _ws_send(websocket,
                                    json.dumps(
                                        {
                                            "status": "error",
                                            "type": "config_error",
                                            "message": "未找到有效的音色模型（请在客户端选择并发送 .pth 文件名）",
                                        }
                                    )
                                )
                                continue

                        if "index_path" in cfg:
                            client_index = os.path.basename(str(cfg.get("index_path") or ""))
                            if not client_index or not (upload_manager.files_dir / client_index).is_file():
                                cfg["index_path"] = ""

                        # 注入 Registry 中的基模路径 (Hubert/RMVPE)
                        try:
                            slots_info = await asyncio.to_thread(model_registry.list_slots)

                            hubert_info = slots_info.get("hubert_base", {})
                            hubert_file = str(hubert_info.get("active", "") or "")
                            hubert_full = upload_manager.files_dir / hubert_file if hubert_file else None
                            cfg["hubert_path"] = hubert_file if hubert_full and hubert_full.is_file() else ""

                            rmvpe_info = slots_info.get("rmvpe", {})
                            rmvpe_file = str(rmvpe_info.get("active", "") or "")
                            rmvpe_full = upload_manager.files_dir / rmvpe_file if rmvpe_file else None
                            cfg["rmvpe_path"] = rmvpe_file if rmvpe_full and rmvpe_full.is_file() else ""
                        except Exception as e:
                            logging.error(f"Error resolving base models: {e}")

                        try:
                            if audio_session is None:
                                raise RuntimeError("audio_endpoint_required")
                            changes = await audio_session.apply_config(cfg)
                        except Exception as e:
                            logging.error(f"Config Error: warmup failed: {e}", exc_info=True)
                            await _ws_send(websocket,
                                json.dumps(
                                    {
                                        "status": "error",
                                        "type": "config_error",
                                        "message": f"模型加载失败：{str(e)}",
                                    }
                                )
                            )
                            continue

                        current_hash = _compute_config_hash(processor.config)

                        effective_block_ms = int(round(processor.core.block_frame * 1000.0 / processor.core.sr))
                        effective_crossfade_ms = int(round(processor.core.crossfade_frame * 1000.0 / processor.core.sr))
                        effective_sola_ms = int(round(processor.core.sola_buffer_frame * 1000.0 / processor.core.sr))
                        response = {
                            "status": "ok",
                            "type": "config_ack",
                            "message": "Config updated",
                            "hash": current_hash,
                            "effective": {
                                "block_ms": effective_block_ms,
                                "crossfade_ms": effective_crossfade_ms,
                                "sola_overlap_ms": effective_sola_ms,
                            },
                        }
                        if isinstance(seq, int):
                            response["seq"] = seq
                        await _ws_send(websocket, json.dumps(response))

                    # 2. 获取日志列表命令
                    elif "command" in data and data["command"] == "list_logs":
                        try:
                            # Use glob to find all files ending in .log
                            log_files = glob.glob(os.path.join(LOG_DIR, "*.log"))

                            # Sort by modification time, newest first
                            log_files.sort(key=os.path.getmtime, reverse=True)

                            # Extract just the filename
                            log_filenames = [os.path.basename(f) for f in log_files]

                            response = {
                                "status": "ok",
                                "type": "log_list",
                                "files": log_filenames,
                                "current": os.path.basename(CURRENT_LOG_FILE)
                            }
                            await _ws_send(websocket, json.dumps(response))
                        except Exception as e:
                            logging.error(f"Error listing logs: {e}", exc_info=True)
                            await _ws_send(websocket, json.dumps({"status": "error", "message": f"List logs error: {str(e)}"}))

                    # 3. 清空历史日志命令
                    elif "command" in data and data["command"] == "clear_old_logs":
                        try:
                            current_basename = os.path.basename(CURRENT_LOG_FILE)
                            log_files = glob.glob(os.path.join(LOG_DIR, "*.log"))
                            deleted = 0
                            for f in log_files:
                                if os.path.basename(f) != current_basename:
                                    try:
                                        os.remove(f)
                                        deleted += 1
                                    except Exception:
                                        pass
                            logging.info(f"Cleared {deleted} old log file(s).")
                            # 返回更新后的日志列表
                            remaining = glob.glob(os.path.join(LOG_DIR, "*.log"))
                            remaining.sort(key=os.path.getmtime, reverse=True)
                            await _ws_send(websocket, json.dumps({
                                "status": "ok",
                                "type": "log_list",
                                "files": [os.path.basename(f) for f in remaining],
                                "current": current_basename,
                            }))
                        except Exception as e:
                            logging.error(f"Error clearing logs: {e}", exc_info=True)
                            await _ws_send(websocket, json.dumps({"status": "error", "message": f"Clear logs error: {str(e)}"}))

                    # 4. 读取日志内容命令
                    elif "command" in data and data["command"] == "read_log":
                        filename = data.get("filename")
                        # 如果没有指定文件名，或文件名是 special token "current"，则读取当前日志
                        target_file = CURRENT_LOG_FILE

                        if filename and filename != "current":
                            # 安全检查：防止路径遍历
                            safe_name = os.path.basename(filename)
                            target_file = os.path.join(LOG_DIR, safe_name)

                        try:
                            if os.path.exists(target_file):
                                with open(target_file, 'r', encoding='utf-8') as f:
                                    content = f.read()

                                response = {
                                    "status": "ok",
                                    "type": "log_content",
                                    "filename": os.path.basename(target_file),
                                    "content": content
                                }
                                await _ws_send(websocket, json.dumps(response))
                            else:
                                logging.error(f"Read Log Error: File not found: {target_file}")
                                await _ws_send(websocket, json.dumps({"status": "error", "message": "File not found"}))
                        except Exception as e:
                            logging.error(f"Error reading log: {e}", exc_info=True)
                            await _ws_send(websocket, json.dumps({"status": "error", "message": f"Read log error: {str(e)}"}))

                    # 5. 实时日志订阅命令
                    elif "command" in data and data["command"] == "watch_log":
                        action = data.get("action")
                        if action == "start":
                            # 1. 发送当前完整日志内容
                            try:
                                if os.path.exists(CURRENT_LOG_FILE):
                                    with open(CURRENT_LOG_FILE, 'r', encoding='utf-8') as f:
                                        content = f.read()
                                    # 发送基础内容
                                    await _ws_send(websocket, json.dumps({
                                        "status": "ok",
                                        "type": "log_content",
                                        "filename": os.path.basename(CURRENT_LOG_FILE),
                                        "content": content
                                    }))
                            except Exception as e:
                                logging.error(f"Error reading initial log: {e}")

                            # 2. 加入订阅列表
                            log_subscribers.add(websocket)
                            await _ws_send(websocket, json.dumps({"status": "ok", "message": "Log watch started"}))

                        elif action == "stop":
                            log_subscribers.discard(websocket)
                            await _ws_send(websocket, json.dumps({"status": "ok", "message": "Log watch stopped"}))

                    # 5. Ping 命令 (用于精确测量 RTT)
                    elif "command" in data and data["command"] == "ping":
                        client_ts = data.get("ts", 0)
                        # 直接回 Pong
                        await _ws_send(websocket, json.dumps({
                            "type": "pong",
                            "client_ts": client_ts,
                            "server_ts": time.perf_counter() * 1000
                        }))

                    elif "command" in data and data["command"] == "files_list":
                        try:
                            files = await asyncio.to_thread(upload_manager.list_files)
                            files = await asyncio.to_thread(
                                _enrich_files_with_voice_meta, files, upload_manager.files_dir
                            )
                            await _ws_send(websocket,
                                json.dumps(
                                    {
                                        "status": "ok",
                                        "type": "files_list",
                                        "files": files,
                                    }
                                )
                            )
                        except Exception as e:
                            logging.error(f"Files List Error: {e}", exc_info=True)
                            await _ws_send(websocket,
                                json.dumps(
                                    {
                                        "status": "error",
                                        "type": "files_error",
                                        "message": str(e),
                                    }
                                )
                            )

                    elif "command" in data and data["command"] == "files_delete":
                        try:
                            name = data.get("name", "")
                            safe_name = os.path.basename(str(name))
                            await asyncio.to_thread(upload_manager.delete_file, name=name)
                            await asyncio.to_thread(model_registry.remove_file_references, filename=safe_name)
                            await _ws_send(websocket, json.dumps({
                                "status": "ok", "type": "files_deleted", "name": safe_name,
                            }))
                            slots = await asyncio.to_thread(model_registry.list_slots)
                            await _ws_send(websocket, json.dumps({
                                "status": "ok", "type": "model_slots", "slots": slots,
                            }))
                            await _send_voice_models()
                        except Exception as e:
                            logging.error(f"Files Delete Error: {e}", exc_info=True)
                            await _ws_send(websocket,
                                json.dumps(
                                    {
                                        "status": "error",
                                        "type": "files_error",
                                        "message": str(e),
                                    }
                                )
                            )

                    elif "command" in data and data["command"] == "files_rename":
                        try:
                            old_name = data.get("old_name", "")
                            new_name = data.get("new_name", "")
                            old_safe = os.path.basename(str(old_name))
                            new_safe = await asyncio.to_thread(
                                upload_manager.rename_file, old_name=old_name, new_name=new_name
                            )
                            await asyncio.to_thread(
                                model_registry.rename_file_references,
                                old_name=old_safe,
                                new_name=new_safe,
                            )

                            await _ws_send(websocket,
                                json.dumps(
                                    {
                                        "status": "ok",
                                        "type": "files_renamed",
                                        "old_name": old_safe,
                                        "new_name": new_safe,
                                    }
                                )
                            )

                            slots = await asyncio.to_thread(model_registry.list_slots)
                            await _ws_send(websocket,
                                json.dumps(
                                    {
                                        "status": "ok",
                                        "type": "model_slots",
                                        "slots": slots,
                                    }
                                )
                            )
                            voice = await asyncio.to_thread(model_registry.list_voice_models)
                            await _ws_send(websocket,
                                json.dumps(
                                    {
                                        "status": "ok",
                                        "type": "voice_models",
                                        "voice": voice,
                                    }
                                )
                            )
                        except Exception as e:
                            logging.error(f"Files Rename Error: {e}", exc_info=True)
                            await _ws_send(websocket,
                                json.dumps(
                                    {
                                        "status": "error",
                                        "type": "files_error",
                                        "message": str(e),
                                    }
                                )
                            )

                    elif "command" in data and data["command"] == "model_list_slots":
                        try:
                            slots = await asyncio.to_thread(model_registry.list_slots)
                            await _ws_send(websocket,
                                json.dumps(
                                    {
                                        "status": "ok",
                                        "type": "model_slots",
                                        "slots": slots,
                                    }
                                )
                            )
                        except Exception as e:
                            logging.error(f"Model List Slots Error: {e}", exc_info=True)
                            await _ws_send(websocket,
                                json.dumps(
                                    {
                                        "status": "error",
                                        "type": "model_error",
                                        "message": str(e),
                                    }
                                )
                            )

                    elif "command" in data and data["command"] == "model_add_to_slot":
                        try:
                            slot = data.get("slot", "")
                            filename = data.get("filename", "")
                            slot_state = await asyncio.to_thread(
                                model_registry.add_to_slot,
                                slot=slot,
                                filename=filename,
                                files_dir=upload_manager.files_dir,
                            )
                            await _ws_send(websocket,
                                json.dumps(
                                    {
                                        "status": "ok",
                                        "type": "model_slot_updated",
                                        "slot": str(slot),
                                        "state": slot_state,
                                    }
                                )
                            )
                        except Exception as e:
                            logging.error(f"Model Add To Slot Error: {e}", exc_info=True)
                            await _ws_send(websocket,
                                json.dumps(
                                    {
                                        "status": "error",
                                        "type": "model_error",
                                        "message": str(e),
                                    }
                                )
                            )

                    elif "command" in data and data["command"] == "model_activate_in_slot":
                        try:
                            slot = data.get("slot", "")
                            filename = data.get("filename", "")
                            slot_state = await asyncio.to_thread(
                                model_registry.activate_in_slot, slot=slot, filename=filename
                            )
                            await _ws_send(websocket,
                                json.dumps(
                                    {
                                        "status": "ok",
                                        "type": "model_slot_updated",
                                        "slot": str(slot),
                                        "state": slot_state,
                                    }
                                )
                            )
                        except Exception as e:
                            logging.error(f"Model Activate In Slot Error: {e}", exc_info=True)
                            await _ws_send(websocket,
                                json.dumps(
                                    {
                                        "status": "error",
                                        "type": "model_error",
                                        "message": str(e),
                                    }
                                )
                            )

                    elif "command" in data and data["command"] == "model_remove_from_slot":
                        try:
                            slot = data.get("slot", "")
                            filename = data.get("filename", "")
                            slot_state = await asyncio.to_thread(
                                model_registry.remove_from_slot, slot=slot, filename=filename
                            )
                            await _ws_send(websocket,
                                json.dumps(
                                    {
                                        "status": "ok",
                                        "type": "model_slot_updated",
                                        "slot": str(slot),
                                        "state": slot_state,
                                    }
                                )
                            )
                        except Exception as e:
                            logging.error(f"Model Remove From Slot Error: {e}", exc_info=True)
                            await _ws_send(websocket,
                                json.dumps(
                                    {
                                        "status": "error",
                                        "type": "model_error",
                                        "message": str(e),
                                    }
                                )
                            )

                    elif "command" in data and data["command"] == "model_set_slot":
                        try:
                            slot = data.get("slot", "")
                            filename = data.get("filename", "")
                            slot_state = await asyncio.to_thread(
                                model_registry.set_slot,
                                slot=slot,
                                filename=filename,
                                files_dir=upload_manager.files_dir,
                            )
                            await _ws_send(websocket,
                                json.dumps(
                                    {
                                        "status": "ok",
                                        "type": "model_slot_updated",
                                        "slot": str(slot),
                                        "state": slot_state,
                                    }
                                )
                            )
                        except Exception as e:
                            logging.error(f"Model Set Slot Error: {e}", exc_info=True)
                            await _ws_send(websocket,
                                json.dumps(
                                    {
                                        "status": "error",
                                        "type": "model_error",
                                        "message": str(e),
                                    }
                                )
                            )

                    elif "command" in data and data["command"] == "voice_model_list":
                        try:
                            await _send_voice_models()
                        except Exception as e:
                            logging.error(f"Voice Model List Error: {e}", exc_info=True)
                            await _ws_send(websocket,
                                json.dumps(
                                    {
                                        "status": "error",
                                        "type": "voice_model_error",
                                        "message": str(e),
                                    }
                                )
                            )

                    elif "command" in data and data["command"] == "voice_model_add":
                        try:
                            name = data.get("name", "")
                            pth = data.get("pth", "")
                            index = data.get("index", "")
                            await asyncio.to_thread(
                                model_registry.add_voice_model,
                                name=name,
                                pth=pth,
                                index=index,
                                files_dir=upload_manager.files_dir,
                            )
                            await _send_voice_models()
                        except Exception as e:
                            logging.error(f"Voice Model Add Error: {e}", exc_info=True)
                            await _ws_send(websocket,
                                json.dumps(
                                    {
                                        "status": "error",
                                        "type": "voice_model_error",
                                        "message": str(e),
                                    }
                                )
                            )

                    elif "command" in data and data["command"] == "voice_model_activate":
                        try:
                            model_id = data.get("id", "")
                            await asyncio.to_thread(
                                model_registry.activate_voice_model, model_id=model_id
                            )
                            await _send_voice_models()
                        except Exception as e:
                            logging.error(f"Voice Model Activate Error: {e}", exc_info=True)
                            await _ws_send(websocket,
                                json.dumps(
                                    {
                                        "status": "error",
                                        "type": "voice_model_error",
                                        "message": str(e),
                                    }
                                )
                            )

                    elif "command" in data and data["command"] == "voice_model_remove":
                        try:
                            model_id = data.get("id", "")
                            await asyncio.to_thread(
                                model_registry.remove_voice_model, model_id=model_id
                            )
                            await _send_voice_models()
                        except Exception as e:
                            logging.error(f"Voice Model Remove Error: {e}", exc_info=True)
                            await _ws_send(websocket,
                                json.dumps(
                                    {
                                        "status": "error",
                                        "type": "voice_model_error",
                                        "message": str(e),
                                    }
                                )
                            )

                    elif "command" in data and data["command"] == "voice_model_preload":
                        try:
                            model_id = str(data.get("id", "") or "").strip()
                            if not model_id:
                                raise ValueError("invalid_model_id")

                            voice = await asyncio.to_thread(model_registry.list_voice_models)
                            models = voice.get("models") if isinstance(voice.get("models"), list) else []
                            model_item = next(
                                (
                                    m
                                    for m in models
                                    if isinstance(m, dict) and str(m.get("id") or "") == model_id
                                ),
                                None,
                            )
                            if model_item is None:
                                raise ValueError("unknown_voice_model")

                            model_pth = _abs_voice_model_path(model_item.get("pth", ""))
                            if not model_pth or not os.path.exists(model_pth):
                                raise FileNotFoundError("file_not_found_pth")

                            slots_info = await asyncio.to_thread(model_registry.list_slots)
                            hubert_file = str(slots_info.get("hubert_base", {}).get("active", "") or "")
                            preload_cfg = {"hubert_path": hubert_file} if hubert_file else {}
                            if preload_cfg:
                                await asyncio.to_thread(processor.update_config, preload_cfg)
                            await asyncio.to_thread(processor.preload_voice_model, model_pth)
                            await _send_voice_models()
                        except Exception as e:
                            logging.error(f"Voice Model Preload Error: {e}", exc_info=True)
                            await _ws_send(websocket,
                                json.dumps(
                                    {
                                        "status": "error",
                                        "type": "voice_model_error",
                                        "message": str(e),
                                    }
                                )
                            )

                    elif "command" in data and data["command"] == "upload_init":
                        try:
                            name = data.get("name", "")
                            size = int(data.get("size", 0))
                            sha256 = data.get("sha256", "")
                            meta = await asyncio.to_thread(
                                upload_manager.init_upload,
                                name=name,
                                size=size,
                                sha256=sha256,
                            )
                            await _ws_send(websocket,
                                json.dumps(
                                    {
                                        "status": "ok",
                                        "type": "upload_ready",
                                        "upload_id": meta.upload_id,
                                        "name": meta.name,
                                        "received_bytes": meta.received_bytes,
                                        "total_bytes": meta.size,
                                    }
                                )
                            )
                        except Exception as e:
                            logging.error(f"Upload Init Error: {e}", exc_info=True)
                            await _ws_send(websocket,
                                json.dumps(
                                    {
                                        "status": "error",
                                        "type": "upload_error",
                                        "message": str(e),
                                    }
                                )
                            )

                    elif "command" in data and data["command"] == "upload_finish":
                        try:
                            upload_id = str(data.get("upload_id", "")).strip()
                            meta, final_name = await asyncio.to_thread(
                                upload_manager.finish_sync, upload_id=upload_id
                            )
                            await _ws_send(websocket,
                                json.dumps(
                                    {
                                        "status": "ok",
                                        "type": "upload_done",
                                        "upload_id": meta.upload_id,
                                        "name": final_name,
                                    }
                                )
                            )
                        except Exception as e:
                            logging.error(f"Upload Finish Error: {e}", exc_info=True)
                            await _ws_send(websocket,
                                json.dumps(
                                    {
                                        "status": "error",
                                        "type": "upload_error",
                                        "message": str(e),
                                    }
                                )
                            )

                    elif "command" in data and data["command"] == "upload_abort":
                        try:
                            upload_id = str(data.get("upload_id", "")).strip()
                            await asyncio.to_thread(upload_manager.abort_sync, upload_id=upload_id)
                            await _ws_send(websocket,
                                json.dumps(
                                    {
                                        "status": "ok",
                                        "type": "upload_aborted",
                                        "upload_id": upload_id,
                                    }
                                )
                            )
                        except Exception as e:
                            logging.error(f"Upload Abort Error: {e}", exc_info=True)
                            await _ws_send(websocket,
                                json.dumps(
                                    {
                                        "status": "error",
                                        "type": "upload_error",
                                        "message": str(e),
                                    }
                                )
                            )

                    else:
                        logging.error(f"Invalid config or command: {data}")
                        response = {"status": "error", "message": "Invalid config or command"}
                        await _ws_send(websocket, json.dumps(response))
                except json.JSONDecodeError as e:
                    logging.error(f"JSON decode error: {e}", exc_info=True)
                    response = {"status": "error", "message": f"JSON decode error: {str(e)}"}
                    await _ws_send(websocket, json.dumps(response))

            elif isinstance(message, bytes):
                if role == "audio":
                    if audio_session is None:
                        continue
                    try:
                        frame = parse_audio_input_frame(message)
                        if frame is None:
                            raise ValueError("invalid_audio_frame_magic")
                        await audio_session.enqueue(frame)
                    except Exception as e:
                        logging.warning(f"Invalid audio frame: {e}")
                        await _ws_send(websocket, json.dumps({
                            "status": "error", "type": "stream_error", "message": str(e)
                        }))
                    continue

                # Control connection binary frames are reserved for resumable file upload.
                try:
                    parsed = parse_file_chunk_frame(message)
                    if parsed is None:
                        raise ValueError("unsupported_control_binary_frame")
                except Exception as e:
                    logging.error(f"Parse File Chunk Error: {e}")
                    await _ws_send(websocket, json.dumps({
                        "status": "error", "type": "upload_error", "message": str(e)
                    }))
                    continue

                upload_uuid, offset, payload = parsed
                upload_id = str(upload_uuid)
                try:
                    meta = await asyncio.to_thread(
                        upload_manager.write_chunk_sync,
                        upload_id=upload_id, offset=int(offset), payload=payload,
                    )
                    await _ws_send(websocket, json.dumps({
                        "status": "ok", "type": "upload_progress",
                        "upload_id": meta.upload_id, "name": meta.name,
                        "received_bytes": meta.received_bytes, "total_bytes": meta.size,
                    }))
                except Exception as e:
                    msg = str(e)
                    if msg.startswith("offset_mismatch:"):
                        expected = int(msg.split(":", 1)[1])
                        await _ws_send(websocket, json.dumps({
                            "status": "error", "type": "upload_offset_mismatch",
                            "upload_id": upload_id, "expected_offset": expected,
                        }))
                    else:
                        logging.error(f"Upload Chunk Write Error: {e}", exc_info=True)
                        await _ws_send(websocket, json.dumps({
                            "status": "error", "type": "upload_error",
                            "upload_id": upload_id, "message": msg,
                        }))

    except websockets.exceptions.ConnectionClosed:
        logging.info("Client disconnected")
    except Exception as e:
        logging.exception(f"Error in binary_echo_handler: {e}")
    finally:
        log_subscribers.discard(websocket)
        if audio_session is not None:
            await audio_session.close()
        await asyncio.to_thread(processor.close)
        _ws_send_locks.pop(websocket, None)

def _preload_base_models() -> None:
    """服务器启动时预加载 Hubert Base（所有音色共用），减少首次推理延迟。"""
    try:
        import torch
        from rvc_infer import _load_hubert
        from pathlib import Path

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        is_half = device.type == "cuda"

        slots_info = model_registry.list_slots()
        files_dir = upload_manager.files_dir

        # 预加载 Hubert Base
        hubert_active = slots_info.get("hubert_base", {}).get("active", "")
        if hubert_active:
            hubert_path = files_dir / hubert_active
            if hubert_path.exists():
                logging.info(f"Preloading Hubert model: {hubert_active}")
                _load_hubert(device, is_half, str(hubert_path))
                logging.info("Hubert model preloaded.")

        # F0 模型（RMVPE / FCPE）不预加载——首次 warmup 时根据客户端
        # 选择的 f0method 懒加载到全局缓存，后续切换音色无需重新加载。

    except Exception as e:
        logging.warning(f"Base model preload failed (non-fatal): {e}")


async def main():
    global _main_loop
    bind_host = os.environ.get("RVC_STREAMING_BIND", "127.0.0.1").strip() or "127.0.0.1"
    bind_port = int(os.environ.get("RVC_STREAMING_PORT", "8765"))
    token = os.environ.get("RVC_STREAMING_TOKEN", "")
    loopback_hosts = {"127.0.0.1", "::1", "localhost"}
    is_loopback = bind_host in loopback_hosts
    if not is_loopback and not token:
        raise RuntimeError(
            "Refusing non-loopback bind without RVC_STREAMING_TOKEN. "
            "Set a strong shared token or bind only to localhost."
        )

    tls_context = None
    cert_path = os.environ.get("RVC_TLS_CERT", "").strip()
    key_path = os.environ.get("RVC_TLS_KEY", "").strip()
    if cert_path or key_path:
        if not cert_path or not key_path:
            raise RuntimeError("Both RVC_TLS_CERT and RVC_TLS_KEY are required for TLS")
        tls_context = ssl.SSLContext(ssl.PROTOCOL_TLS_SERVER)
        tls_context.minimum_version = ssl.TLSVersion.TLSv1_2
        tls_context.load_cert_chain(certfile=cert_path, keyfile=key_path)
    elif not is_loopback and os.environ.get("RVC_ALLOW_INSECURE_WS", "").strip() != "1":
        raise RuntimeError(
            "Refusing clear-text remote WebSocket. Configure RVC_TLS_CERT/RVC_TLS_KEY, "
            "or set RVC_ALLOW_INSECURE_WS=1 only on a trusted private network."
        )

    loop = asyncio.get_running_loop()
    _main_loop = loop
    loop.run_in_executor(None, _preload_base_models)

    broadcaster_task = asyncio.create_task(log_broadcaster())
    try:
        async with websockets.serve(
            binary_echo_handler,
            bind_host,
            bind_port,
            max_size=2 * 1024 * 1024,
            max_queue=4,
            compression=None,
            ping_interval=20,
            ping_timeout=20,
            ssl=tls_context,
        ):
            logging.info(
                f"RVC Server listening on {'wss' if tls_context else 'ws'}://{bind_host}:{bind_port} "
                f"(auth={'enabled' if token else 'local-only'})"
            )
            await asyncio.get_running_loop().create_future()
    finally:
        broadcaster_task.cancel()
        with contextlib.suppress(asyncio.CancelledError):
            await broadcaster_task

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        logging.info("Server stopped by user via KeyboardInterrupt")
    except Exception:
        logging.exception("Fatal error in main server execution")
