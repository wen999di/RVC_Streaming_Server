import asyncio
import contextlib
import websockets
import logging
import hashlib
from logging.handlers import RotatingFileHandler
import struct
import time
import json
import os
import glob
from pathlib import Path
from collections.abc import Mapping
from rvc_core import RVCCore
from file_transfer import UploadManager, parse_file_chunk_frame
from model_registry import ModelRegistry

# 全局状态
log_subscribers = set()
log_queue = asyncio.Queue()
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
        "crossfade_length", "extra_time", "stream_chunk_ms", 
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
        cpt = torch.load(str(path), map_location="cpu")
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

class WebSocketLogHandler(logging.Handler):
    """将日志写入 asyncio 队列，用于广播到 WebSocket 订阅者。"""
    def emit(self, record):
        try:
            msg = self.format(record)
            # 使用 call_soon_threadsafe 入队，避免跨线程直接操作 asyncio.Queue。
            try:
                loop = asyncio.get_running_loop()
                loop.call_soon_threadsafe(log_queue.put_nowait, msg + "\n")
            except RuntimeError:
                # 启动/关闭阶段可能没有运行中的 loop。
                pass
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

async def log_broadcaster():
    """后台任务：将日志广播给订阅者。"""
    while True:
        msg = await log_queue.get()
        if log_subscribers:
            # 广播给所有订阅者
            for ws in list(log_subscribers):
                try:
                    await ws.send(json.dumps({"status": "ok", "type": "log_chunk", "content": msg}))
                except Exception:
                    log_subscribers.discard(ws)
        log_queue.task_done()


class AudioProcessor:
    def __init__(self, config=None):
        self.config = config or {}
        self.core = RVCCore(_resolve_runtime_config(self.config))
        logging.info(f"AudioProcessor initialized with config: {self.config}")

    def update_config(self, config):
        self.config.update(config)
        self.core.update_config(_resolve_runtime_config(self.config))
        logging.info(f"Config updated: {self.config}")
 
    def warmup(self):
        return self.core.warmup()

    def process_frame(self, audio_data, ts_start_ns=None):
        return self.core.process_frame(audio_data, ts_start_ns)

async def binary_echo_handler(websocket):
    logging.info(f"Client connected: {websocket.remote_address}")

    # 1. 为连接初始化音频处理器(等待客户端配置)
    processor = AudioProcessor()
    logging.info("Audio Processor Initialized (waiting for client config)")

    loop = asyncio.get_running_loop()
    # 出站队列容量 35（约 700ms@20ms/chunk），用于吸收接收速率波动。
    # outgoing_queue: (proc_time_ms, enqueue_time_s, ts_ns, audio_chunk)
    outgoing_queue: asyncio.Queue[tuple] = asyncio.Queue(maxsize=35)

    async def sender_loop():
        next_deadline = time.perf_counter()
        
        while True:
            stream_chunk_ms = int(processor.config.get("stream_chunk_ms", 20) or 20)
            if stream_chunk_ms <= 0:
                stream_chunk_ms = 20
            interval_s = stream_chunk_ms / 1000.0

            # 策略：队列积压较大时开启“追赶模式”（Burst Mode）。
            # 阈值设为 约 100ms 的数据量
            burst_threshold = max(1, int(100 / stream_chunk_ms))
            q_size = outgoing_queue.qsize()
            
            now = time.perf_counter()
            
            # 如果积压 > 阈值，跳过 sleep 直接发送，直到积压消除
            if q_size <= burst_threshold:
                if now < next_deadline:
                    await asyncio.sleep(next_deadline - now)
                
                # 只有在极大延迟（如断网重连后）才重置 deadline，避免时间漂移
                # 允许 1秒 的时间债，超过 1秒 才重置，保证长期平均速率匹配
                if now > next_deadline + 1.0:
                    next_deadline = now
                
                next_deadline += interval_s
            else:
                # Burst mode: 不等待，快速消耗队列以降低延迟
                # 让出一点 CPU 时间片
                await asyncio.sleep(0)
                # 在追赶模式下，我们将 deadline 重置为当前，以便追赶结束后重新按节奏发送
                next_deadline = time.perf_counter() + interval_s

            try:
                item = outgoing_queue.get_nowait()
                proc_time, enqueue_time, ts_ns, payload_chunk = item
                
                # 计算队列等待时间
                queue_wait_ms = int((time.perf_counter() - enqueue_time) * 1000)
                if queue_wait_ms > 65535: queue_wait_ms = 65535
                
                header = struct.pack('>HHQ', proc_time, queue_wait_ms, int(ts_ns))
                final_payload = header + payload_chunk
                
            except asyncio.QueueEmpty:
                continue

            try:
                await websocket.send(final_payload)
            except Exception as e:
                logging.error(f"发送失败: {e}")
            finally:
                outgoing_queue.task_done()

    sender_task = asyncio.create_task(sender_loop())
    last_backlog_log_ts = 0.0
    empty_out_streak = 0
    last_empty_out_log_ts = 0.0
    try:
        async for message in websocket:
            # 根据消息类型进行处理
            if isinstance(message, str):
                # JSON 配置消息或命令
                try:
                    data = json.loads(message)
                    
                    # 1. 配置更新
                    if "config" in data:
                        cfg = data["config"] if isinstance(data.get("config"), dict) else {}
                        seq = data.get("seq", None)
                        passthrough = bool(cfg.get("passthrough", False))
                        
                        # Handle Model Path Resolution if 'model_path' is provided
                        if "model_path" in cfg and not passthrough:
                            client_pth = os.path.basename(str(cfg.get("model_path") or ""))
                            if client_pth:
                                candidate_pth = upload_manager.files_dir / client_pth
                                if candidate_pth.exists() and candidate_pth.is_file():
                                    cfg["model_path"] = client_pth

                            if not cfg.get("model_path"):
                                logging.error(f"Config Error: Voice model not found. client_pth={client_pth}")
                                await websocket.send(
                                    json.dumps(
                                        {
                                            "status": "error",
                                            "type": "config_error",
                                            "message": "未找到有效的音色模型（请在客户端选择并发送 .pth 文件名）",
                                        }
                                    )
                                )
                                continue
                        elif "model_path" in cfg and passthrough:
                            cfg["model_path"] = os.path.basename(str(cfg.get("model_path") or "")) if cfg.get("model_path") else ""
                        
                        # Handle Index Path Resolution if 'index_path' is provided
                        if "index_path" in cfg:
                            client_index = os.path.basename(str(cfg.get("index_path") or ""))
                            if client_index:
                                candidate_index = upload_manager.files_dir / client_index
                                if candidate_index.exists() and candidate_index.is_file():
                                    cfg["index_path"] = client_index
                                else:
                                    cfg["index_path"] = ""
                            else:
                                cfg["index_path"] = ""

                        # 注入 Registry 中的基模路径 (Hubert/RMVPE)
                        try:
                            slots_info = await asyncio.to_thread(model_registry.list_slots)
                            
                            hubert_info = slots_info.get("hubert_base", {})
                            hubert_file = hubert_info.get("active", "")
                            if hubert_file:
                                hubert_full = upload_manager.files_dir / hubert_file
                                if hubert_full.exists():
                                    cfg["hubert_path"] = hubert_file
                            
                            rmvpe_info = slots_info.get("rmvpe", {})
                            rmvpe_file = rmvpe_info.get("active", "")
                            if rmvpe_file:
                                rmvpe_full = upload_manager.files_dir / rmvpe_file
                                if rmvpe_full.exists():
                                    cfg["rmvpe_path"] = rmvpe_file
                        except Exception as e:
                            logging.error(f"Error resolving base models: {e}")

                        try:
                            processor.update_config(cfg)
                            if not bool(getattr(processor.core, "passthrough", False)) and processor.core.model_path:
                                await asyncio.to_thread(processor.warmup)
                        except Exception as e:
                            logging.error(f"Config Error: warmup failed: {e}", exc_info=True)
                            await websocket.send(
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
                        response = {
                            "status": "ok",
                            "type": "config_ack",
                            "message": "Config updated",
                            "hash": current_hash,
                            "effective": {
                                "block_ms": effective_block_ms,
                                "crossfade_ms": effective_crossfade_ms,
                            },
                        }
                        if isinstance(seq, int):
                            response["seq"] = seq
                        await websocket.send(json.dumps(response))
                    
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
                            await websocket.send(json.dumps(response))
                        except Exception as e:
                            logging.error(f"Error listing logs: {e}", exc_info=True)
                            await websocket.send(json.dumps({"status": "error", "message": f"List logs error: {str(e)}"}))

                    # 3. 读取日志内容命令
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
                                await websocket.send(json.dumps(response))
                            else:
                                logging.error(f"Read Log Error: File not found: {target_file}")
                                await websocket.send(json.dumps({"status": "error", "message": "File not found"}))
                        except Exception as e:
                            logging.error(f"Error reading log: {e}", exc_info=True)
                            await websocket.send(json.dumps({"status": "error", "message": f"Read log error: {str(e)}"}))
                            
                    # 4. 实时日志订阅命令
                    elif "command" in data and data["command"] == "watch_log":
                        action = data.get("action")
                        if action == "start":
                            # 1. 发送当前完整日志内容
                            try:
                                if os.path.exists(CURRENT_LOG_FILE):
                                    with open(CURRENT_LOG_FILE, 'r', encoding='utf-8') as f:
                                        content = f.read()
                                    # 发送基础内容
                                    await websocket.send(json.dumps({
                                        "status": "ok",
                                        "type": "log_content",
                                        "filename": os.path.basename(CURRENT_LOG_FILE),
                                        "content": content
                                    }))
                            except Exception as e:
                                logging.error(f"Error reading initial log: {e}")
                            
                            # 2. 加入订阅列表
                            log_subscribers.add(websocket)
                            await websocket.send(json.dumps({"status": "ok", "message": "Log watch started"}))
                            
                        elif action == "stop":
                            log_subscribers.discard(websocket)
                            await websocket.send(json.dumps({"status": "ok", "message": "Log watch stopped"}))
                            
                    # 5. Ping 命令 (用于精确测量 RTT)
                    elif "command" in data and data["command"] == "ping":
                        client_ts = data.get("ts", 0)
                        # 直接回 Pong
                        await websocket.send(json.dumps({
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
                            await websocket.send(
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
                            await websocket.send(
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
                            await asyncio.to_thread(upload_manager.delete_file, name=name)
                            await websocket.send(
                                json.dumps(
                                    {
                                        "status": "ok",
                                        "type": "files_deleted",
                                        "name": os.path.basename(str(name)),
                                    }
                                )
                            )
                        except Exception as e:
                            logging.error(f"Files Delete Error: {e}", exc_info=True)
                            await websocket.send(
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

                            await websocket.send(
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
                            await websocket.send(
                                json.dumps(
                                    {
                                        "status": "ok",
                                        "type": "model_slots",
                                        "slots": slots,
                                    }
                                )
                            )
                            voice = await asyncio.to_thread(model_registry.list_voice_models)
                            await websocket.send(
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
                            await websocket.send(
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
                            await websocket.send(
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
                            await websocket.send(
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
                            await websocket.send(
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
                            await websocket.send(
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
                            await websocket.send(
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
                            await websocket.send(
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
                            await websocket.send(
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
                            await websocket.send(
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
                            await websocket.send(
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
                            await websocket.send(
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
                            voice = await asyncio.to_thread(model_registry.list_voice_models)
                            await websocket.send(
                                json.dumps(
                                    {
                                        "status": "ok",
                                        "type": "voice_models",
                                        "voice": voice,
                                    }
                                )
                            )
                        except Exception as e:
                            logging.error(f"Voice Model List Error: {e}", exc_info=True)
                            await websocket.send(
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
                            voice = await asyncio.to_thread(
                                model_registry.add_voice_model,
                                name=name,
                                pth=pth,
                                index=index,
                                files_dir=upload_manager.files_dir,
                            )
                            await websocket.send(
                                json.dumps(
                                    {
                                        "status": "ok",
                                        "type": "voice_models",
                                        "voice": voice,
                                    }
                                )
                            )
                        except Exception as e:
                            logging.error(f"Voice Model Add Error: {e}", exc_info=True)
                            await websocket.send(
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
                            voice = await asyncio.to_thread(
                                model_registry.activate_voice_model, model_id=model_id
                            )
                            await websocket.send(
                                json.dumps(
                                    {
                                        "status": "ok",
                                        "type": "voice_models",
                                        "voice": voice,
                                    }
                                )
                            )
                        except Exception as e:
                            logging.error(f"Voice Model Activate Error: {e}", exc_info=True)
                            await websocket.send(
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
                            voice = await asyncio.to_thread(
                                model_registry.remove_voice_model, model_id=model_id
                            )
                            await websocket.send(
                                json.dumps(
                                    {
                                        "status": "ok",
                                        "type": "voice_models",
                                        "voice": voice,
                                    }
                                )
                            )
                        except Exception as e:
                            logging.error(f"Voice Model Remove Error: {e}", exc_info=True)
                            await websocket.send(
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
                            await websocket.send(
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
                            await websocket.send(
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
                            await websocket.send(
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
                            await websocket.send(
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
                            await websocket.send(
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
                            await websocket.send(
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
                        await websocket.send(json.dumps(response))
                except json.JSONDecodeError as e:
                    logging.error(f"JSON decode error: {e}", exc_info=True)
                    response = {"status": "error", "message": f"JSON decode error: {str(e)}"}
                    await websocket.send(json.dumps(response))
                    
            elif isinstance(message, bytes):
                try:
                    parsed = parse_file_chunk_frame(message)
                except Exception as e:
                    logging.error(f"Parse File Chunk Error: {e}", exc_info=True)
                    await websocket.send(
                        json.dumps(
                            {
                                "status": "error",
                                "type": "upload_error",
                                "message": str(e),
                            }
                        )
                    )
                    continue

                if parsed is not None:
                    upload_uuid, offset, payload = parsed
                    upload_id = str(upload_uuid)
                    try:
                        meta = await asyncio.to_thread(
                            upload_manager.write_chunk_sync,
                            upload_id=upload_id,
                            offset=int(offset),
                            payload=payload,
                        )
                        await websocket.send(
                            json.dumps(
                                {
                                    "status": "ok",
                                    "type": "upload_progress",
                                    "upload_id": meta.upload_id,
                                    "name": meta.name,
                                    "received_bytes": meta.received_bytes,
                                    "total_bytes": meta.size,
                                }
                            )
                        )
                    except Exception as e:
                        msg = str(e)
                        if msg.startswith("offset_mismatch:"):
                            expected = int(msg.split(":", 1)[1])
                            logging.warning(f"Upload Offset Mismatch: upload_id={upload_id}, expected={expected}")
                            await websocket.send(
                                json.dumps(
                                    {
                                        "status": "error",
                                        "type": "upload_offset_mismatch",
                                        "upload_id": upload_id,
                                        "expected_offset": expected,
                                    }
                                )
                            )
                        else:
                            logging.error(f"Upload Chunk Write Error: {e}", exc_info=True)
                            await websocket.send(
                                json.dumps(
                                    {
                                        "status": "error",
                                        "type": "upload_error",
                                        "upload_id": upload_id,
                                        "message": msg,
                                    }
                                )
                            )
                    continue

                if len(message) < 8:
                    continue
                ts_start_ns = struct.unpack(">Q", message[:8])[0]
                audio_payload = message[8:]

                def _process_with_timing(pcm_bytes, ts_ns):
                    t0 = time.perf_counter()
                    out_pcm, out_ts_ns = processor.process_frame(pcm_bytes, ts_ns)
                    proc_ms = int(round((time.perf_counter() - t0) * 1000.0))
                    return out_pcm, out_ts_ns, proc_ms

                processed_audio, out_start_ns, proc_time_ms = await loop.run_in_executor(
                    None, _process_with_timing, audio_payload, ts_start_ns
                )

                if not processed_audio:
                    empty_out_streak += 1
                    now = time.perf_counter()
                    if now - last_empty_out_log_ts > 2.0 and empty_out_streak >= 20:
                        logging.warning(
                            f"输出为空（连续 {empty_out_streak} 次），block_frame={int(processor.core.block_frame)} pending={int(processor.core.pending_samples)}"
                        )
                        last_empty_out_log_ts = now
                    continue
                empty_out_streak = 0

                stream_chunk_ms = int(processor.config.get("stream_chunk_ms", 20) or 20)
                if stream_chunk_ms <= 0:
                    stream_chunk_ms = 20
                
                # Calculate bytes based on output sampling rate
                output_sr = getattr(processor.core, "output_sr", 16000)
                bytes_per_sample = getattr(processor.core, "bytes_per_sample", 4)
                bytes_per_ms = int(output_sr * bytes_per_sample / 1000)
                chunk_bytes = max(bytes_per_sample, stream_chunk_ms * bytes_per_ms)
                chunk_bytes = (chunk_bytes // bytes_per_sample) * bytes_per_sample

                for offset in range(0, len(processed_audio), chunk_bytes):
                    chunk = processed_audio[offset : offset + chunk_bytes]
                    out_proc = proc_time_ms
                    offset_samples = offset // bytes_per_sample
                    
                    # Timestamp calculation depends on output_sr
                    ns_per_sample = 1_000_000_000 / output_sr
                    chunk_ts_ns = int(out_start_ns or 0) + int(offset_samples * ns_per_sample)
                    
                    # 记录入队时间，用于计算在输出队列中的等待时间
                    enqueue_time = time.perf_counter()
                    
                    # 队列满时丢弃旧包
                    while outgoing_queue.full():
                        try:
                            outgoing_queue.get_nowait()
                            outgoing_queue.task_done()
                        except asyncio.QueueEmpty:
                            break
                    try:
                        outgoing_queue.put_nowait((out_proc, enqueue_time, chunk_ts_ns, chunk))
                    except asyncio.QueueFull:
                        pass
                
                # 队列积压监控（阈值调整为 25，容量 35 的 ~70%）
                queue_size = outgoing_queue.qsize()
                if queue_size > 25:
                    now = time.perf_counter()
                    if now - last_backlog_log_ts > 5.0:
                        logging.warning(f"输出队列积压: {queue_size} 包")
                        last_backlog_log_ts = now
                
    except websockets.exceptions.ConnectionClosed:
        logging.info("Client disconnected")
    except Exception as e:
        logging.exception(f"Error in binary_echo_handler: {e}")
    finally:
        log_subscribers.discard(websocket)
        sender_task.cancel()

async def main():
    # max_size=None 允许大包传输
    broadcaster_task = asyncio.create_task(log_broadcaster())
    try:
        async with websockets.serve(binary_echo_handler, "0.0.0.0", 8765, max_size=None):
            logging.info("Binary RVC Server running on :8765")
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
