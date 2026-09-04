import asyncio
import argparse
import contextlib
import errno
import websockets
import logging
import hashlib
import hmac
from logging.handlers import RotatingFileHandler
import time
import threading
import signal
import ssl
import json
import os
import glob
import struct
import sys
import uuid
from pathlib import Path
from collections.abc import Mapping
from rvc_core import RVCCore
from audio_protocol import (
    AudioInputFrame, FLAG_DISCONTINUITY, build_audio_output_frame, parse_audio_input_frame,
)
from file_transfer import UploadManager, parse_file_chunk_frame, sanitize_relative_path
from hub_download import HubDownloadManager
from model_registry import ModelRegistry
from training_manager import TrainingManager

# 全局状态
log_subscribers = set()
_ws_send_locks = {}
log_queue = asyncio.Queue(maxsize=1000)
_main_loop = None
_server_source_dir = Path(__file__).resolve().parent
_data_dir = Path(os.environ.get("RVC_DATA_DIR") or _server_source_dir).resolve()
_data_dir.mkdir(parents=True, exist_ok=True)
upload_manager = UploadManager(_data_dir)
hub_download_manager = HubDownloadManager(upload_manager.files_dir)
model_registry = ModelRegistry(_data_dir)

_STDIO_MAGIC = b"RVCP"
_STDIO_HEADER = struct.Struct("<4sBBI")
_STDIO_CHANNEL_CONTROL = 0
_STDIO_CHANNEL_AUDIO = 1
_STDIO_CHANNEL_TRANSPORT = 255
_STDIO_KIND_TEXT = 1
_STDIO_KIND_BINARY = 2
_STDIO_KIND_CLOSE = 3
_STDIO_KIND_READY = 4
_STDIO_MAX_MESSAGE_BYTES = 2 * 1024 * 1024


def _read_exactly(stream, length: int) -> bytes:
    chunks = bytearray()
    while len(chunks) < length:
        chunk = stream.read(length - len(chunks))
        if not chunk:
            if not chunks:
                return b""
            raise EOFError("stdio frame ended before the declared length")
        chunks.extend(chunk)
    return bytes(chunks)


def _read_stdio_frame(stream):
    header = _read_exactly(stream, _STDIO_HEADER.size)
    if not header:
        return None
    magic, channel, kind, payload_length = _STDIO_HEADER.unpack(header)
    if magic != _STDIO_MAGIC:
        raise ValueError("invalid stdio transport magic")
    if payload_length > _STDIO_MAX_MESSAGE_BYTES:
        raise ValueError(f"stdio frame is too large: {payload_length}")
    payload = _read_exactly(stream, payload_length)
    if len(payload) != payload_length:
        raise EOFError("stdio frame payload ended unexpectedly")
    return channel, kind, payload


class _StdioFrameWriter:
    def __init__(self, stream):
        self._stream = stream
        self._lock = asyncio.Lock()

    async def send(self, channel: int, kind: int, payload: bytes = b"") -> None:
        if len(payload) > _STDIO_MAX_MESSAGE_BYTES:
            raise ValueError(f"stdio frame is too large: {len(payload)}")
        frame = _STDIO_HEADER.pack(_STDIO_MAGIC, channel, kind, len(payload))
        async with self._lock:
            self._stream.write(frame)
            if payload:
                self._stream.write(payload)
            self._stream.flush()


class _StdioEndpoint:
    def __init__(self, channel: int, path: str, writer: _StdioFrameWriter):
        self.channel = channel
        self.path = path
        self.request_headers = {}
        self.remote_address = "parent-process"
        self.closed = False
        self._writer = writer
        self._messages = asyncio.Queue(maxsize=16 if channel == _STDIO_CHANNEL_AUDIO else 64)

    def __aiter__(self):
        return self

    async def __anext__(self):
        message = await self._messages.get()
        if message is None:
            raise StopAsyncIteration
        return message

    async def send(self, payload) -> None:
        if self.closed:
            raise ConnectionError("stdio endpoint is closed")
        if isinstance(payload, str):
            kind = _STDIO_KIND_TEXT
            data = payload.encode("utf-8")
        else:
            kind = _STDIO_KIND_BINARY
            data = bytes(payload)
        await self._writer.send(self.channel, kind, data)

    async def close(self, code=1000, reason="") -> None:
        if self.closed:
            return
        self.closed = True
        await self._writer.send(self.channel, _STDIO_KIND_CLOSE)
        self._feed_close()

    def feed(self, kind: int, payload: bytes) -> None:
        if self.closed:
            return
        if kind == _STDIO_KIND_CLOSE:
            self._feed_close()
            return
        if kind == _STDIO_KIND_TEXT:
            message = payload.decode("utf-8")
        elif kind == _STDIO_KIND_BINARY:
            message = payload
        else:
            raise ValueError(f"unsupported stdio frame kind: {kind}")
        try:
            self._messages.put_nowait(message)
        except asyncio.QueueFull:
            if self.channel != _STDIO_CHANNEL_AUDIO:
                raise
            # Realtime input is latest-wins, matching the WebSocket queue policy.
            with contextlib.suppress(asyncio.QueueEmpty):
                self._messages.get_nowait()
                self._messages.task_done()
            self._messages.put_nowait(message)

    def _feed_close(self) -> None:
        if self.closed and self._messages.empty():
            with contextlib.suppress(asyncio.QueueFull):
                self._messages.put_nowait(None)
            return
        self.closed = True
        while True:
            try:
                self._messages.get_nowait()
                self._messages.task_done()
            except asyncio.QueueEmpty:
                break
        with contextlib.suppress(asyncio.QueueFull):
            self._messages.put_nowait(None)


def _register_completed_training(job: dict) -> None:
    model_file = str(job.get("model_file") or "")
    index_file = str(job.get("index_file") or "")
    if not model_file:
        return
    base_name = str(job.get("name") or Path(model_file).stem)
    speakers = job.get("speaker_outputs") if isinstance(job.get("speaker_outputs"), list) else []
    if not speakers:
        speakers = [{"id": 0, "name": "", "index_file": index_file}]
    for speaker in speakers:
        speaker_id = int(speaker.get("id", 0))
        speaker_name = str(speaker.get("name") or "").strip()
        model_registry.add_voice_model(
            name=f"{base_name} · {speaker_name}" if len(speakers) > 1 and speaker_name else base_name,
            pth=model_file,
            index=str(speaker.get("index_file") or ""),
            speaker_id=speaker_id,
            files_dir=upload_manager.files_dir,
        )


training_manager = TrainingManager(
    base_dir=_data_dir,
    files_dir=upload_manager.files_dir,
    on_complete=_register_completed_training,
)

_base_model_preload_state_lock = threading.Lock()
_base_model_preload_running = False
_base_model_preload_pending = False

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
        "model_path", "index_path", "speaker_id", "f0_up_key", "block_time",
        "crossfade_length", "extra_time", "stream_chunk_ms",
        "formant_shift", "f0method", "index_rate", "passthrough",
        "silence_db_threshold", "silence_gate_atten",
        "input_noise_reduce", "output_noise_reduce", "noise_reduce_prop_decrease",
        "rms_mix_rate"
    ]

    float_keys = {
        "block_time", "crossfade_length", "extra_time",
        "formant_shift", "index_rate", "speaker_id",
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


def _required_base_model_slot_error(config: Mapping) -> str | None:
    if bool(config.get("passthrough", False)) or not str(config.get("model_path") or "").strip():
        return None

    missing = []
    if not str(config.get("hubert_path") or "").strip():
        missing.append("HuBERT Base")
    if (
        str(config.get("f0method") or "rmvpe").strip().lower() == "rmvpe"
        and not str(config.get("rmvpe_path") or "").strip()
    ):
        missing.append("RMVPE")

    if len(missing) == 2:
        return "未配置 HuBERT Base 和 RMVPE 模型槽位"
    if missing:
        return f"未配置 {missing[0]} 模型槽位"
    return None


def _clamp_stream_chunk_ms(value) -> int:
    try:
        chunk_ms = int(round(float(value)))
    except (TypeError, ValueError):
        chunk_ms = 20
    return max(10, min(120, chunk_ms))


def _iter_output_slices(payload: bytes, timestamp_ns: int, chunk_ms: int, sample_rate: int = 16000):
    """Yield paced packet payloads without changing the inference block size."""
    bytes_per_sample = 4
    total_samples = len(payload) // bytes_per_sample
    if total_samples <= 0:
        return

    chunk_samples = max(1, int(round(sample_rate * _clamp_stream_chunk_ms(chunk_ms) / 1000.0)))
    ns_per_sample = 1_000_000_000 // sample_rate
    base_timestamp_ns = max(0, int(timestamp_ns or 0))
    for sample_offset in range(0, total_samples, chunk_samples):
        sample_count = min(chunk_samples, total_samples - sample_offset)
        byte_offset = sample_offset * bytes_per_sample
        byte_count = sample_count * bytes_per_sample
        slice_timestamp_ns = (
            base_timestamp_ns + sample_offset * ns_per_sample
            if base_timestamp_ns > 0
            else 0
        )
        yield payload[byte_offset : byte_offset + byte_count], slice_timestamp_ns, sample_count


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
LOG_DIR = str(_data_dir / "logs")
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

    def prepare(self):
        with self._lock:
            return self.core.prepare()

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
            base_ts_ns = int(ts_start_ns or 0)
            results = []
            for offset in range(0, len(audio_data), block_bytes):
                chunk = audio_data[offset : offset + block_bytes]
                offset_samples = offset // int(self.core.bytes_per_sample)
                chunk_ts = base_ts_ns + offset_samples * ns_per_sample if base_ts_ns > 0 else 0
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
        # Configuration belongs to this audio connection only.  A reconnect
        # must provide a fresh complete configuration before streaming starts.
        self.configuration_received = False
        self.active_session_id = 0
        self.last_input_sequence = None
        self.output_sequence = 0
        self.output_slice_ms = _clamp_stream_chunk_ms(os.environ.get("RVC_AUDIO_OUTPUT_SLICE_MS", "20"))
        self._next_output_send_time = 0.0
        self._output_epoch = 0
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
            self._output_epoch += 1
            self._next_output_send_time = 0.0
            self._pending_discontinuity = True
            self._output_discontinuity_pending = True
            self._drain_input()
            self._drain(self.output_queue)
            await asyncio.to_thread(self.processor.reset_stream_state)

    async def apply_config(self, cfg: dict) -> dict:
        async with self._state_lock:
            requested_slice_ms = _clamp_stream_chunk_ms(
                cfg.get("stream_chunk_ms", self.output_slice_ms)
            )
            pacing_changed = requested_slice_ms != self.output_slice_ms
            self.output_slice_ms = requested_slice_ms

            changes = await asyncio.to_thread(self.processor.update_config, cfg)
            reset_timeline = (
                changes.get("buffer_layout")
                or changes.get("model_runtime")
                or pacing_changed
            )
            if reset_timeline:
                self.last_input_sequence = None
                self.output_sequence = 0
                self._output_epoch += 1
                self._next_output_send_time = 0.0
                self._pending_discontinuity = True
                self._output_discontinuity_pending = True
                self._drain_input()
                self._drain(self.output_queue)
                await asyncio.to_thread(self.processor.reset_stream_state)
            changes["output_pacing"] = pacing_changed
            should_warmup = bool(changes.get("model_runtime") or changes.get("buffer_layout"))
            if should_warmup and not self.processor.core.passthrough and self.processor.core.model_path:
                # Complete the one-time, real-shape CUDA warmup while the UI is
                # still showing "loading", not on the first live audio block.
                await asyncio.to_thread(self.processor.warmup)
            self.configuration_received = True
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

                    if dropped_output:
                        # Abort any older block that the paced sender is still
                        # emitting. Latest-wins must apply at slice granularity.
                        self._output_epoch += 1
                    flags = FLAG_DISCONTINUITY if (self._output_discontinuity_pending or dropped_output) else 0
                    self._output_discontinuity_pending = False
                    item = (
                        frame.session_id, self._output_epoch, int(out_ts_ns or 0), proc_ms,
                        input_queue_ms, flags, out_pcm, time.perf_counter(),
                    )
                    self.output_queue.put_nowait(item)
            finally:
                self.input_queue.task_done()

    async def _sender_loop(self) -> None:
        loop = asyncio.get_running_loop()
        while True:
            item = await self.output_queue.get()
            try:
                session_id, output_epoch, ts_ns, proc_ms, input_queue_ms, flags, payload, enqueue_time = item
                if session_id != self.active_session_id or output_epoch != self._output_epoch:
                    continue

                # Measure only the time the completed block actually waited in
                # the output queue. Reusing this value for every paced slice
                # avoids counting the intentional 20 ms send schedule as queue
                # congestion (which previously produced a 0..250 ms sawtooth).
                output_queue_ms = int(round((time.perf_counter() - enqueue_time) * 1000.0))
                first_slice = True
                for slice_payload, slice_ts_ns, slice_samples in _iter_output_slices(
                    payload, ts_ns, self.output_slice_ms
                ):
                    if session_id != self.active_session_id or output_epoch != self._output_epoch:
                        break

                    slice_duration_s = slice_samples / 16000.0
                    now = loop.time()
                    late_tolerance_s = max(0.002, slice_duration_s * 0.25)
                    if (
                        self._next_output_send_time <= 0.0
                        or now - self._next_output_send_time > late_tolerance_s
                    ):
                        # Never catch up by sending a burst after an inference or
                        # event-loop stall; restart pacing from the current clock.
                        self._next_output_send_time = now

                    delay_s = self._next_output_send_time - now
                    if delay_s > 0.0:
                        await asyncio.sleep(delay_s)
                    if session_id != self.active_session_id or output_epoch != self._output_epoch:
                        break

                    sequence = self.output_sequence
                    self.output_sequence = (self.output_sequence + 1) & 0xFFFFFFFF
                    frame = build_audio_output_frame(
                        session_id=session_id, sequence=sequence, sample_rate=16000,
                        timestamp_ns=slice_ts_ns, proc_ms=proc_ms, input_queue_ms=input_queue_ms,
                        output_queue_ms=output_queue_ms,
                        flags=flags if first_slice else 0,
                        payload=slice_payload,
                    )
                    await _ws_send(self.websocket, frame)
                    first_slice = False
                    self._next_output_send_time += slice_duration_s
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
    hub_download_tasks: dict[str, tuple[asyncio.Task, threading.Event]] = {}
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

    async def _run_hub_download(request_id: str, cancel_event: threading.Event, args: dict) -> None:
        loop = asyncio.get_running_loop()

        def send_progress(payload: dict) -> None:
            message = json.dumps({"status": "ok", "type": "hub_download_progress", **payload})

            def schedule() -> None:
                if not websocket.closed:
                    asyncio.create_task(_ws_send(websocket, message))

            try:
                loop.call_soon_threadsafe(schedule)
            except RuntimeError:
                pass

        def download_worker() -> None:
            try:
                result = hub_download_manager.download_selected(
                    request_id=request_id,
                    provider=args.get("provider"),
                    repo=args.get("repo"),
                    revision=args.get("revision"),
                    paths=args.get("paths") or [],
                    destination=args.get("destination"),
                    cancel_event=cancel_event,
                    progress=send_progress,
                )
            except BaseException as exc:
                try:
                    loop.call_soon_threadsafe(complete_future, None, exc)
                except RuntimeError:
                    pass
            else:
                try:
                    loop.call_soon_threadsafe(complete_future, result, None)
                except RuntimeError:
                    pass

        completion = loop.create_future()

        def complete_future(result, error) -> None:
            if completion.done():
                return
            if error is not None:
                completion.set_exception(error)
            else:
                completion.set_result(result)

        threading.Thread(
            target=download_worker,
            name=f"rvc-hub-download-{request_id[:8]}",
            daemon=True,
        ).start()

        try:
            result = await completion
            await _ws_send(websocket, json.dumps({
                "status": "ok",
                "type": "hub_download_done",
                "request_id": result.request_id,
                "provider": result.provider,
                "repo": result.repo_id,
                "revision": result.revision,
                "destination": result.destination,
                "files": list(result.files),
                "total_bytes": result.total_bytes,
            }))
        except InterruptedError:
            if not websocket.closed:
                await _ws_send(websocket, json.dumps({
                    "status": "ok", "type": "hub_download_cancelled", "request_id": request_id,
                }))
        except asyncio.CancelledError:
            cancel_event.set()
            raise
        except Exception as exc:
            logging.error("Hub download failed: %s", exc, exc_info=True)
            if not websocket.closed:
                await _ws_send(websocket, json.dumps({
                    "status": "error",
                    "type": "hub_download_error",
                    "request_id": request_id,
                    "message": str(exc),
                }))
        finally:
            hub_download_tasks.pop(request_id, None)

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
                        if training_manager.is_active():
                            await _ws_send(websocket, json.dumps({
                                "status": "error", "type": "stream_error",
                                "message": "训练任务运行期间暂停实时变声",
                            }))
                            continue
                        if not audio_session.configuration_received:
                            await _ws_send(websocket, json.dumps({
                                "status": "error", "type": "config_required",
                                "message": "请先发送本次连接的参数配置",
                            }))
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
                        logging.info("Config request: seq=%s keys=%s", seq, sorted(cfg.keys()))
                        if not cfg:
                            await _ws_send(websocket, json.dumps({
                                "status": "error", "type": "config_error",
                                "message": "参数配置不能为空",
                            }))
                            continue
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
                            cfg["hubert_path"] = ""
                            cfg["rmvpe_path"] = ""

                        effective_cfg = dict(processor.config)
                        effective_cfg.update(cfg)
                        slot_error = _required_base_model_slot_error(effective_cfg)
                        if slot_error:
                            logging.warning("Config rejected: %s", slot_error)
                            await _ws_send(
                                websocket,
                                json.dumps(
                                    {
                                        "status": "error",
                                        "type": "config_error",
                                        "message": slot_error,
                                    }
                                ),
                            )
                            continue

                        try:
                            if audio_session is None:
                                raise RuntimeError("audio_endpoint_required")
                            apply_started = time.perf_counter()
                            changes = await audio_session.apply_config(cfg)
                            logging.info(
                                "Config applied: seq=%s model_runtime=%s buffer_layout=%s elapsed=%.1fms",
                                seq,
                                bool(changes.get("model_runtime")),
                                bool(changes.get("buffer_layout")),
                                (time.perf_counter() - apply_started) * 1000.0,
                            )
                        except Exception as e:
                            logging.error(f"Config Error: model prepare failed: {e}", exc_info=True)
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
                        effective_stream_chunk_ms = int(audio_session.output_slice_ms)
                        response = {
                            "status": "ok",
                            "type": "config_ack",
                            "message": "Config updated",
                            "hash": current_hash,
                            "effective": {
                                "block_ms": effective_block_ms,
                                "crossfade_ms": effective_crossfade_ms,
                                "sola_overlap_ms": effective_sola_ms,
                                "stream_chunk_ms": effective_stream_chunk_ms,
                                "cuda_graph": processor.core.cuda_graph_status(),
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
                            safe_name = sanitize_relative_path(str(name))
                            await asyncio.to_thread(upload_manager.delete_file, name=name)
                            await asyncio.to_thread(model_registry.remove_file_references, filename=safe_name)
                            _schedule_base_model_preload()
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
                            old_safe = sanitize_relative_path(str(old_name))
                            new_safe = await asyncio.to_thread(
                                upload_manager.rename_file, old_name=old_name, new_name=new_name
                            )
                            await asyncio.to_thread(
                                model_registry.rename_file_references,
                                old_name=old_safe,
                                new_name=new_safe,
                            )
                            _schedule_base_model_preload()

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

                    elif "command" in data and data["command"] == "training_organize_files":
                        try:
                            result = await asyncio.to_thread(
                                upload_manager.organize_training_files,
                                model_name=data.get("model_name", ""),
                                files=data.get("files", []),
                            )
                            for moved in result["files"]:
                                await asyncio.to_thread(
                                    model_registry.rename_file_references,
                                    old_name=moved["old_name"],
                                    new_name=moved["new_name"],
                                )
                            await _ws_send(websocket, json.dumps({
                                "status": "ok",
                                "type": "training_files_organized",
                                **result,
                            }, ensure_ascii=False))
                        except Exception as e:
                            logging.error("Training Files Organize Error: %s", e, exc_info=True)
                            await _ws_send(websocket, json.dumps({
                                "status": "error",
                                "type": "training_organize_error",
                                "message": str(e),
                            }, ensure_ascii=False))

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
                            if str(slot) in ("hubert_base", "rmvpe"):
                                _schedule_base_model_preload()
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
                            if str(slot) in ("hubert_base", "rmvpe"):
                                _schedule_base_model_preload()
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
                            if str(slot) in ("hubert_base", "rmvpe"):
                                _schedule_base_model_preload()
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
                            if str(slot) in ("hubert_base", "rmvpe"):
                                _schedule_base_model_preload()
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

                    elif "command" in data and data["command"] == "training_list":
                        try:
                            training = await asyncio.to_thread(training_manager.snapshot)
                            await _ws_send(
                                websocket,
                                json.dumps({"status": "ok", "type": "training_jobs", "training": training}),
                            )
                        except Exception as e:
                            logging.error("Training List Error: %s", e, exc_info=True)
                            await _ws_send(
                                websocket,
                                json.dumps({"status": "error", "type": "training_error", "message": str(e)}),
                            )

                    elif "command" in data and data["command"] == "training_start":
                        try:
                            slots = await asyncio.to_thread(model_registry.list_slots)
                            request = dict(data.get("training") or {})
                            request["hubert"] = str(slots.get("hubert_base", {}).get("active") or "")
                            request["rmvpe"] = str(slots.get("rmvpe", {}).get("active") or "")
                            request["pymss_weight"] = str(slots.get("pymss_weight", {}).get("active") or "")
                            request["pymss_config"] = str(slots.get("pymss_config", {}).get("active") or "")
                            use_pretrained = request.get("use_pretrained") is True
                            request["use_pretrained"] = use_pretrained
                            request["pretrained_g"] = (
                                str(slots.get("pretrained_g", {}).get("active") or "")
                                if use_pretrained else ""
                            )
                            request["pretrained_d"] = (
                                str(slots.get("pretrained_d", {}).get("active") or "")
                                if use_pretrained else ""
                            )
                            job = await asyncio.to_thread(training_manager.start, request)
                            await _ws_send(
                                websocket,
                                json.dumps({"status": "ok", "type": "training_started", "job": job}),
                            )
                            training = await asyncio.to_thread(training_manager.snapshot)
                            await _ws_send(
                                websocket,
                                json.dumps({"status": "ok", "type": "training_jobs", "training": training}),
                            )
                        except Exception as e:
                            logging.error("Training Start Error: %s", e, exc_info=True)
                            await _ws_send(
                                websocket,
                                json.dumps({"status": "error", "type": "training_error", "message": str(e)}),
                            )

                    elif "command" in data and data["command"] == "training_cancel":
                        try:
                            job = await asyncio.to_thread(training_manager.cancel, data.get("id", ""))
                            await _ws_send(
                                websocket,
                                json.dumps({"status": "ok", "type": "training_cancelled", "job": job}),
                            )
                        except Exception as e:
                            logging.error("Training Cancel Error: %s", e, exc_info=True)
                            await _ws_send(
                                websocket,
                                json.dumps({"status": "error", "type": "training_error", "message": str(e)}),
                            )

                    elif "command" in data and data["command"] == "training_delete":
                        try:
                            await asyncio.to_thread(training_manager.delete, data.get("id", ""))
                            training = await asyncio.to_thread(training_manager.snapshot)
                            await _ws_send(
                                websocket,
                                json.dumps({"status": "ok", "type": "training_jobs", "training": training}),
                            )
                        except Exception as e:
                            logging.error("Training Delete Error: %s", e, exc_info=True)
                            await _ws_send(
                                websocket,
                                json.dumps({"status": "error", "type": "training_error", "message": str(e)}),
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
                            speaker_id = data.get("speaker_id", 0)
                            await asyncio.to_thread(
                                model_registry.add_voice_model,
                                name=name,
                                pth=pth,
                                index=index,
                                speaker_id=speaker_id,
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
                            hubert_full = upload_manager.files_dir / hubert_file if hubert_file else None
                            if not hubert_full or not hubert_full.is_file():
                                await _ws_send(
                                    websocket,
                                    json.dumps(
                                        {
                                            "status": "error",
                                            "type": "voice_model_error",
                                            "message": "未配置 HuBERT Base 模型槽位",
                                        }
                                    ),
                                )
                                continue
                            await asyncio.to_thread(processor.update_config, {"hubert_path": hubert_file})
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

                    elif "command" in data and data["command"] == "hub_repo_list":
                        try:
                            repository = await asyncio.to_thread(
                                hub_download_manager.list_repository,
                                provider=data.get("provider"),
                                repo=data.get("repo"),
                                revision=data.get("revision", ""),
                            )
                            await _ws_send(websocket, json.dumps({
                                "status": "ok",
                                "type": "hub_repo_files",
                                "provider": repository.provider,
                                "repo": repository.repo_id,
                                "revision": repository.revision,
                                "default_destination": hub_download_manager.default_destination(
                                    repository.provider, repository.repo_id
                                ),
                                "total_bytes": repository.total_size,
                                "files": [item.to_dict() for item in repository.files],
                            }))
                        except Exception as exc:
                            logging.error("Hub repository list error: %s", exc, exc_info=True)
                            await _ws_send(websocket, json.dumps({
                                "status": "error",
                                "type": "hub_repo_error",
                                "message": str(exc),
                            }))

                    elif "command" in data and data["command"] == "hub_download_start":
                        try:
                            request_id = str(data.get("request_id") or "").strip().lower()
                            if not request_id:
                                raise ValueError("缺少下载任务 ID")
                            try:
                                request_id = str(uuid.UUID(request_id))
                            except ValueError as exc:
                                raise ValueError("无效的下载任务 ID") from exc
                            if request_id in hub_download_tasks:
                                raise ValueError("下载任务已存在")
                            if len(hub_download_tasks) >= 2:
                                raise ValueError("同时最多执行两个仓库下载任务")
                            cancel_event = threading.Event()
                            task = asyncio.create_task(_run_hub_download(request_id, cancel_event, dict(data)))
                            hub_download_tasks[request_id] = (task, cancel_event)
                            await _ws_send(websocket, json.dumps({
                                "status": "ok",
                                "type": "hub_download_started",
                                "request_id": request_id,
                            }))
                        except Exception as exc:
                            await _ws_send(websocket, json.dumps({
                                "status": "error",
                                "type": "hub_download_error",
                                "request_id": str(data.get("request_id") or ""),
                                "message": str(exc),
                            }))

                    elif "command" in data and data["command"] == "hub_download_cancel":
                        request_id = str(data.get("request_id") or "").strip().lower()
                        current = hub_download_tasks.get(request_id)
                        if current is not None:
                            current[1].set()
                        else:
                            await _ws_send(websocket, json.dumps({
                                "status": "ok",
                                "type": "hub_download_cancelled",
                                "request_id": request_id,
                            }))

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
                    if not audio_session.configuration_received:
                        continue
                    if training_manager.is_active():
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
        pending_hub_tasks = list(hub_download_tasks.values())
        for task, cancel_event in pending_hub_tasks:
            cancel_event.set()
            task.cancel()
        if pending_hub_tasks:
            await asyncio.gather(
                *(task for task, _ in pending_hub_tasks),
                return_exceptions=True,
            )
        if audio_session is not None:
            await audio_session.close()
        await asyncio.to_thread(processor.close)
        _ws_send_locks.pop(websocket, None)

def _active_base_model_paths() -> dict[str, Path]:
    """返回用户明确设置且文件仍存在的基模槽位。"""
    slots_info = model_registry.list_slots()
    files_dir = upload_manager.files_dir
    active_paths: dict[str, Path] = {}

    for slot in ("hubert_base", "rmvpe"):
        active = str(slots_info.get(slot, {}).get("active") or "").strip()
        if not active:
            continue

        safe_name = os.path.basename(active)
        if safe_name != active:
            logging.warning("Ignoring invalid active filename in %s slot: %s", slot, active)
            continue

        path = files_dir / safe_name
        if not path.is_file():
            logging.warning("Skipping base model preload; %s slot file is missing: %s", slot, safe_name)
            continue
        active_paths[slot] = path

    return active_paths


def _preload_base_models() -> None:
    """只预加载用户明确设置的 HuBERT/RMVPE 槽位。"""
    try:
        active_paths = _active_base_model_paths()
    except Exception as e:
        logging.warning("Unable to read base model slots for preload (non-fatal): %s", e)
        return

    if not active_paths:
        logging.info("Base model preload skipped: no HuBERT/RMVPE slot is configured.")
        return

    try:
        import torch
        from rvc_infer import _device_infer_lock, _load_hubert, _load_rmvpe

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        is_half = device.type == "cuda"
        loaders = (
            ("hubert_base", "HuBERT Base", _load_hubert),
            ("rmvpe", "RMVPE", _load_rmvpe),
        )

        # 与实时推理共用同一设备锁，避免 CUDA 初始化或显存分配互相争用。
        with _device_infer_lock(device):
            for slot, title, loader in loaders:
                path = active_paths.get(slot)
                if path is None:
                    continue
                started = time.perf_counter()
                try:
                    logging.info("Preloading %s from configured slot: %s", title, path.name)
                    loader(device, is_half, str(path))
                    logging.info(
                        "%s preloaded in %.1f ms: %s",
                        title,
                        (time.perf_counter() - started) * 1000.0,
                        path.name,
                    )
                except Exception as e:
                    logging.warning("%s preload failed (non-fatal): %s", title, e, exc_info=True)
    except Exception as e:
        logging.warning("Base model preload failed (non-fatal): %s", e, exc_info=True)


def _base_model_preload_worker() -> None:
    """合并短时间内的重复请求，并保证槽位变化后最终预热最新状态。"""
    global _base_model_preload_running, _base_model_preload_pending

    while True:
        with _base_model_preload_state_lock:
            _base_model_preload_pending = False

        try:
            _preload_base_models()
        except Exception:
            logging.exception("Unexpected base model preload worker failure")

        with _base_model_preload_state_lock:
            if _base_model_preload_pending:
                continue
            _base_model_preload_running = False
            return


def _schedule_base_model_preload(loop=None) -> bool:
    """在后台请求一次基模预热；已有任务时将请求合并为一次后续刷新。"""
    global _base_model_preload_running, _base_model_preload_pending

    target_loop = loop or _main_loop
    if target_loop is None or target_loop.is_closed():
        logging.warning("Base model preload was not scheduled because the server loop is unavailable.")
        return False

    with _base_model_preload_state_lock:
        _base_model_preload_pending = True
        if _base_model_preload_running:
            return False
        _base_model_preload_running = True

    try:
        # 不使用 asyncio 的默认线程池。asyncio.run() 退出时会等待默认
        # 线程池中的任务结束；模型加载若恰好卡在驱动层，会让 Ctrl+C 看起来
        # 像没有生效。守护线程不会阻止服务器进程退出。
        threading.Thread(
            target=_base_model_preload_worker,
            name="rvc-base-model-preload",
            daemon=True,
        ).start()
        return True
    except Exception:
        with _base_model_preload_state_lock:
            _base_model_preload_running = False
            _base_model_preload_pending = False
        logging.exception("Failed to schedule base model preload")
        return False


def _is_address_in_use_error(error: OSError) -> bool:
    """同时识别 POSIX EADDRINUSE 与 Windows WSAEADDRINUSE。"""
    return (
        getattr(error, "errno", None) in {errno.EADDRINUSE, 10048}
        or getattr(error, "winerror", None) == 10048
    )


async def main() -> int:
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
    stop_event = asyncio.Event()
    shutdown_requested = False
    previous_signal_handlers = {}

    def request_shutdown(signum, _frame) -> None:
        nonlocal shutdown_requested
        if shutdown_requested:
            return
        shutdown_requested = True
        signal_name = getattr(signum, "name", None) or str(signum)
        logging.info("Received %s; stopping RVC Server...", signal_name)
        loop.call_soon_threadsafe(stop_event.set)

    for signal_name in ("SIGINT", "SIGTERM", "SIGBREAK"):
        shutdown_signal = getattr(signal, signal_name, None)
        if shutdown_signal is None:
            continue
        try:
            previous_signal_handlers[shutdown_signal] = signal.signal(
                shutdown_signal,
                request_shutdown,
            )
        except (ValueError, OSError):
            # signal.signal 只能在主线程安装；嵌入式运行和部分测试环境会走这里。
            continue

    broadcaster_task = None
    try:
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
                close_timeout=3,
                ssl=tls_context,
            ):
                broadcaster_task = asyncio.create_task(log_broadcaster())
                logging.info(
                    f"RVC Server listening on {'wss' if tls_context else 'ws'}://{bind_host}:{bind_port} "
                    f"(auth={'enabled' if token else 'local-only'})"
                )
                # 先成功占用端口，再加载基模。这样重复启动时不会先占用显存、
                # 等待数秒后才报告端口冲突。
                _schedule_base_model_preload(loop)
                await stop_event.wait()
        except OSError as error:
            if not _is_address_in_use_error(error):
                raise
            logging.error(
                "无法启动 RVC Server：%s:%d 已被占用。请先关闭正在运行的服务器，"
                "或通过 RVC_STREAMING_PORT 使用其他端口。",
                bind_host,
                bind_port,
            )
            return 1
        return 0
    finally:
        if broadcaster_task is not None:
            broadcaster_task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await broadcaster_task
        training_manager.shutdown()
        _main_loop = None
        for shutdown_signal, previous_handler in previous_signal_handlers.items():
            with contextlib.suppress(ValueError, OSError):
                signal.signal(shutdown_signal, previous_handler)


async def main_stdio() -> int:
    """Serve both logical endpoints over the parent/child stdio pipes."""
    global _main_loop
    loop = asyncio.get_running_loop()
    _main_loop = loop
    stop_event = asyncio.Event()
    protocol_stream = sys.stdout.buffer
    # Keep stdout exclusively binary-framed. Any legacy print() call is routed
    # to stderr, which the desktop client consumes as human-readable log text.
    sys.stdout = sys.stderr
    writer = _StdioFrameWriter(protocol_stream)
    endpoints = {
        _STDIO_CHANNEL_CONTROL: _StdioEndpoint(
            _STDIO_CHANNEL_CONTROL, "/control", writer
        ),
        _STDIO_CHANNEL_AUDIO: _StdioEndpoint(
            _STDIO_CHANNEL_AUDIO, "/audio", writer
        ),
    }

    def stop_from_reader(error=None) -> None:
        if error is not None:
            logging.error("Local stdio transport stopped: %s", error)
        for endpoint in endpoints.values():
            endpoint._feed_close()
        stop_event.set()

    def read_frames() -> None:
        try:
            while True:
                frame = _read_stdio_frame(sys.stdin.buffer)
                if frame is None:
                    loop.call_soon_threadsafe(stop_from_reader)
                    return
                channel, kind, payload = frame
                endpoint = endpoints.get(channel)
                if endpoint is None:
                    raise ValueError(f"unsupported stdio channel: {channel}")
                loop.call_soon_threadsafe(endpoint.feed, kind, payload)
        except BaseException as exc:
            loop.call_soon_threadsafe(stop_from_reader, exc)

    reader_thread = threading.Thread(
        target=read_frames,
        name="rvc-stdio-reader",
        daemon=True,
    )
    endpoint_tasks = [
        asyncio.create_task(binary_echo_handler(endpoints[_STDIO_CHANNEL_CONTROL])),
        asyncio.create_task(binary_echo_handler(endpoints[_STDIO_CHANNEL_AUDIO])),
    ]
    broadcaster_task = asyncio.create_task(log_broadcaster())
    reader_thread.start()

    try:
        await asyncio.sleep(0)
        await writer.send(_STDIO_CHANNEL_TRANSPORT, _STDIO_KIND_READY)
        logging.info("RVC Server connected through private parent-process stdio pipes")
        _schedule_base_model_preload(loop)
        await stop_event.wait()
        return 0
    finally:
        for endpoint in endpoints.values():
            endpoint._feed_close()
        try:
            await asyncio.wait_for(
                asyncio.gather(*endpoint_tasks, return_exceptions=True),
                timeout=5.0,
            )
        except asyncio.TimeoutError:
            for task in endpoint_tasks:
                task.cancel()
            await asyncio.gather(*endpoint_tasks, return_exceptions=True)
        broadcaster_task.cancel()
        with contextlib.suppress(asyncio.CancelledError):
            await broadcaster_task
        training_manager.shutdown()
        _main_loop = None


def _parse_args():
    parser = argparse.ArgumentParser(description="RVC Streaming Server")
    parser.add_argument(
        "--stdio",
        action="store_true",
        help="use private parent-process stdio pipes instead of listening on a TCP port",
    )
    return parser.parse_args()

if __name__ == "__main__":
    try:
        args = _parse_args()
        raise SystemExit(asyncio.run(main_stdio() if args.stdio else main()))
    except KeyboardInterrupt:
        logging.info("Server stopped by user via KeyboardInterrupt")
    except Exception:
        logging.exception("Fatal error in main server execution")
