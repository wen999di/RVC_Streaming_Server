from __future__ import annotations

import json
import os
import subprocess
import sys
import threading
import time
import uuid
from pathlib import Path
from typing import Callable, Mapping

from file_transfer import resolve_relative_file


_TERMINAL_STATES = {"completed", "failed", "cancelled"}
_AUDIO_SUFFIXES = (".wav", ".flac", ".mp3", ".m4a", ".ogg", ".opus")
_PREPROCESS_MODES = {
    "none": ("", ""),
    "vocals": ("bs_roformer", "vocals"),
    "noreverb": ("mel_band_roformer", "noreverb"),
    "karaoke": ("mel_band_roformer", "karaoke"),
}


def _atomic_json(path: Path, value: object) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    with open(tmp, "w", encoding="utf-8") as handle:
        json.dump(value, handle, ensure_ascii=False, indent=2)
    os.replace(tmp, path)


def _bounded_int(value, default: int, low: int, high: int) -> int:
    try:
        parsed = int(value)
    except (TypeError, ValueError):
        parsed = default
    return max(low, min(high, parsed))


class TrainingManager:
    """Runs one isolated RVC training worker at a time and persists its state."""

    def __init__(
        self,
        *,
        base_dir: Path,
        files_dir: Path,
        on_complete: Callable[[dict], None] | None = None,
    ) -> None:
        self.base_dir = Path(base_dir).resolve()
        self.files_dir = Path(files_dir).resolve()
        self.jobs_dir = self.base_dir / "training_jobs"
        self.state_path = self.base_dir / "training_jobs.json"
        self.jobs_dir.mkdir(parents=True, exist_ok=True)
        self._on_complete = on_complete
        self._lock = threading.RLock()
        self._jobs: dict[str, dict] = {}
        self._processes: dict[str, subprocess.Popen] = {}
        self._worker_job_ids: set[str] = set()
        self._load()

    def _load(self) -> None:
        if not self.state_path.exists():
            return
        try:
            with open(self.state_path, "r", encoding="utf-8") as handle:
                raw = json.load(handle)
        except Exception:
            return
        jobs = raw.get("jobs", {}) if isinstance(raw, dict) else {}
        if not isinstance(jobs, dict):
            return
        for job_id, item in jobs.items():
            if not isinstance(item, dict):
                continue
            state = str(item.get("state") or "failed")
            if state not in _TERMINAL_STATES:
                item["state"] = "interrupted"
                item["message"] = "服务器重启，任务可重新开始"
                item["updated_at"] = time.time()
            self._jobs[str(job_id)] = item
        self._save_locked()

    def _save_locked(self) -> None:
        _atomic_json(self.state_path, {"version": 1, "jobs": self._jobs})

    def _public_job(self, job: Mapping) -> dict:
        keys = (
            "id", "name", "state", "stage", "progress", "message",
            "created_at", "updated_at", "file_count", "sample_rate", "epochs",
            "batch_size", "preprocess", "speaker_count", "epoch", "step", "loss",
            "model_file", "index_file", "speaker_outputs", "error",
        )
        return {key: job.get(key) for key in keys if key in job}

    def snapshot(self) -> dict:
        with self._lock:
            jobs = [self._public_job(item) for item in self._jobs.values()]
        jobs.sort(key=lambda item: float(item.get("created_at") or 0), reverse=True)
        active = next(
            (item["id"] for item in jobs if item.get("state") in {"queued", "running", "cancelling"}),
            "",
        )
        return {"jobs": jobs, "active_id": active}

    def is_active(self) -> bool:
        with self._lock:
            return any(
                item.get("state") in {"queued", "running", "cancelling"}
                for item in self._jobs.values()
            )

    def _resolve_input(self, name: object, *, required: bool, suffixes: tuple[str, ...]) -> str:
        raw = str(name or "").strip()
        if not raw:
            if required:
                raise ValueError("缺少训练输入文件")
            return ""
        raw, path = resolve_relative_file(self.files_dir, raw)
        if not path.is_file():
            raise FileNotFoundError(raw)
        if path.suffix.lower() not in suffixes:
            raise ValueError(f"不支持的训练文件类型: {raw}")
        return str(path)

    def _resolve_audio_files(self, value: object) -> list[dict]:
        if not isinstance(value, list) or not value:
            raise ValueError("请至少选择一个训练音频文件")
        if len(value) > 20000:
            raise ValueError("训练音频文件数量不能超过 20000")
        resolved: list[dict] = []
        seen: set[str] = set()
        for raw in value:
            item = raw if isinstance(raw, Mapping) else {"name": raw}
            name = str(item.get("name") or "").strip()
            if not name or name.lower() in seen:
                if name:
                    raise ValueError(f"训练音频重复: {name}")
                raise ValueError("训练音频文件名不能为空")
            name, path = resolve_relative_file(self.files_dir, name)
            if not path.is_file():
                raise FileNotFoundError(name)
            if path.suffix.lower() not in _AUDIO_SUFFIXES:
                raise ValueError(f"不支持的训练音频类型: {name}")
            speaker = str(item.get("speaker") or "speaker").strip()
            if not speaker or len(speaker) > 80:
                raise ValueError(f"说话人名称无效: {name}")
            seen.add(name.lower())
            resolved.append({"name": name, "path": str(path), "speaker": speaker})
        return resolved

    def _resolve_pymss_python(self) -> str:
        candidates: list[Path] = []
        configured = os.environ.get("RVC_PYMSS_PYTHON", "").strip()
        if configured:
            candidates.append(Path(configured).expanduser())
        environment_dir = self.base_dir / ".pixi" / "envs" / "pymss"
        candidates.extend(
            (
                environment_dir / "python.exe",
                environment_dir / "Scripts" / "python.exe",
                environment_dir / "bin" / "python",
            )
        )
        for candidate in candidates:
            if candidate.is_file():
                return str(candidate.resolve())
        raise RuntimeError(
            "尚未安装 PyMSS 训练前处理环境；请在服务器目录运行 pixi install -e pymss，"
            "或设置 RVC_PYMSS_PYTHON"
        )

    def start(self, request: Mapping) -> dict:
        with self._lock:
            if any(
                item.get("state") in {"queued", "running", "cancelling"}
                for item in self._jobs.values()
            ):
                raise RuntimeError("已有训练任务正在运行")

            name = str(request.get("name") or "").strip()
            if not name or len(name) > 80:
                raise ValueError("模型名称不能为空且不能超过 80 个字符")
            audio_files = self._resolve_audio_files(request.get("files"))
            if "use_pretrained" in request:
                use_pretrained = request.get("use_pretrained") is True
            else:
                use_pretrained = bool(request.get("pretrained_g") or request.get("pretrained_d"))
            if use_pretrained and (not request.get("pretrained_g") or not request.get("pretrained_d")):
                raise ValueError("已选择使用预训练权重，但生成器或判别器槽位尚未配置")
            pretrained_g = self._resolve_input(
                request.get("pretrained_g") if use_pretrained else "",
                required=use_pretrained,
                suffixes=(".pth", ".pt"),
            )
            pretrained_d = self._resolve_input(
                request.get("pretrained_d") if use_pretrained else "",
                required=use_pretrained,
                suffixes=(".pth", ".pt"),
            )
            hubert_path = self._resolve_input(
                request.get("hubert"), required=True, suffixes=(".pth", ".pt")
            )
            rmvpe_path = self._resolve_input(
                request.get("rmvpe"), required=True, suffixes=(".pth", ".pt")
            )
            preprocess = str(request.get("preprocess") or "none").strip().lower()
            if preprocess not in _PREPROCESS_MODES:
                raise ValueError("不支持的训练前处理方式")
            pymss_model_type, pymss_stem = _PREPROCESS_MODES[preprocess]
            pymss_weight_path = self._resolve_input(
                request.get("pymss_weight"),
                required=preprocess != "none",
                suffixes=(".ckpt", ".pth", ".th"),
            )
            pymss_config_path = self._resolve_input(
                request.get("pymss_config"),
                required=preprocess != "none",
                suffixes=(".yaml", ".yml"),
            )
            pymss_python = self._resolve_pymss_python() if preprocess != "none" else ""
            sample_rate = _bounded_int(request.get("sample_rate"), 40000, 32000, 48000)
            if sample_rate not in (32000, 40000, 48000):
                raise ValueError("采样率只支持 32000、40000 或 48000")

            now = time.time()
            job_id = uuid.uuid4().hex
            work_dir = (self.jobs_dir / job_id).resolve()
            work_dir.mkdir(parents=True, exist_ok=False)
            job = {
                "id": job_id,
                "name": name,
                "state": "queued",
                "stage": "queued",
                "progress": 0.0,
                "message": "等待训练进程启动",
                "created_at": now,
                "updated_at": now,
                "file_count": len(audio_files),
                "sample_rate": sample_rate,
                "epochs": _bounded_int(request.get("epochs"), 200, 1, 5000),
                "batch_size": _bounded_int(request.get("batch_size"), 4, 1, 64),
                "preprocess": preprocess,
                "use_pretrained": use_pretrained,
                "work_dir": str(work_dir),
            }
            config = {
                **job,
                "audio_files": audio_files,
                "hubert_path": hubert_path,
                "rmvpe_path": rmvpe_path,
                "pymss_python": pymss_python,
                "pymss_weight_path": pymss_weight_path,
                "pymss_config_path": pymss_config_path,
                "pymss_model_type": pymss_model_type,
                "pymss_stem": pymss_stem,
                "pretrained_g": pretrained_g,
                "pretrained_d": pretrained_d,
                "files_dir": str(self.files_dir),
                "version": "v2",
                "save_every": _bounded_int(request.get("save_every"), 25, 1, 1000),
                "learning_rate": float(request.get("learning_rate") or 0.0001),
            }
            config_path = work_dir / "job.json"
            _atomic_json(config_path, config)
            self._jobs[job_id] = job
            self._worker_job_ids.add(job_id)
            self._save_locked()
            thread = threading.Thread(
                target=self._run_job_entry,
                args=(job_id, config_path),
                name=f"rvc-training-{job_id[:8]}",
                daemon=True,
            )
            thread.start()
            return self._public_job(job)

    def cancel(self, job_id: object) -> dict:
        key = str(job_id or "").strip()
        with self._lock:
            job = self._jobs.get(key)
            if job is None:
                raise KeyError("unknown_training_job")
            if job.get("state") in _TERMINAL_STATES:
                return self._public_job(job)
            job["state"] = "cancelling"
            job["message"] = "正在停止训练"
            job["updated_at"] = time.time()
            process = self._processes.get(key)
            self._save_locked()
        if process is not None and process.poll() is None:
            self._terminate_process_tree(process)
        return self._public_job(job)

    def shutdown(self) -> None:
        """Stop worker process trees so a server shutdown cannot leave GPU jobs orphaned."""
        with self._lock:
            active_processes = list(self._processes.items())
            owned_job_ids = set(self._worker_job_ids)
            now = time.time()
            changed = False
            for job_id in owned_job_ids:
                job = self._jobs.get(job_id)
                if job is None:
                    continue
                if job.get("state") in _TERMINAL_STATES or job.get("state") == "interrupted":
                    continue
                job["state"] = "interrupted"
                job["stage"] = "interrupted"
                job["message"] = "服务器已停止，训练任务已中断"
                job["updated_at"] = now
                changed = True
            if changed:
                self._save_locked()

        for _job_id, process in active_processes:
            if process.poll() is None:
                self._terminate_process_tree(process)

        with self._lock:
            self._processes.clear()
            self._worker_job_ids.clear()

    @staticmethod
    def _terminate_process_tree(process: subprocess.Popen) -> None:
        if process.poll() is not None:
            return
        if os.name == "nt":
            subprocess.run(
                ["taskkill", "/PID", str(process.pid), "/T", "/F"],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                check=False,
                creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
            )
            return
        try:
            os.killpg(os.getpgid(process.pid), 15)
        except (AttributeError, ProcessLookupError):
            process.terminate()

    def delete(self, job_id: object) -> None:
        key = str(job_id or "").strip()
        with self._lock:
            job = self._jobs.get(key)
            if job is None:
                raise KeyError("unknown_training_job")
            if job.get("state") not in _TERMINAL_STATES and job.get("state") != "interrupted":
                raise RuntimeError("运行中的任务不能删除")
            del self._jobs[key]
            self._save_locked()

    def _apply_event(self, job_id: str, event: Mapping) -> None:
        with self._lock:
            job = self._jobs.get(job_id)
            if job is None:
                return
            for key in (
                "state", "stage", "progress", "message", "speaker_count",
                "epoch", "step", "loss", "model_file", "index_file", "speaker_outputs", "error",
            ):
                if key in event:
                    job[key] = event[key]
            job["updated_at"] = time.time()
            self._save_locked()

    def _run_job_entry(self, job_id: str, config_path: Path) -> None:
        try:
            self._run_job(job_id, config_path)
        finally:
            with self._lock:
                self._worker_job_ids.discard(job_id)

    def _run_job(self, job_id: str, config_path: Path) -> None:
        command = [sys.executable, "-u", str(Path(__file__).with_name("training_worker.py")), str(config_path)]
        creationflags = getattr(subprocess, "CREATE_NO_WINDOW", 0)
        if os.name == "nt":
            creationflags |= getattr(subprocess, "CREATE_NEW_PROCESS_GROUP", 0)
        with self._lock:
            initial_state = self._jobs.get(job_id, {}).get("state")
            if initial_state == "cancelling":
                self._apply_event(
                    job_id,
                    {"state": "cancelled", "stage": "cancelled", "message": "训练已取消"},
                )
                return
            if initial_state in _TERMINAL_STATES or initial_state == "interrupted":
                return
        try:
            process = subprocess.Popen(
                command,
                cwd=str(Path(__file__).parent),
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                encoding="utf-8",
                errors="replace",
                bufsize=1,
                creationflags=creationflags,
                start_new_session=os.name != "nt",
            )
        except Exception as error:
            self._apply_event(
                job_id,
                {
                    "state": "failed",
                    "stage": "failed",
                    "message": "训练进程启动失败",
                    "error": f"{type(error).__name__}: {error}",
                },
            )
            return
        with self._lock:
            self._processes[job_id] = process
            cancelling = self._jobs.get(job_id, {}).get("state") == "cancelling"
        if cancelling:
            self._terminate_process_tree(process)
        else:
            self._apply_event(job_id, {"state": "running", "stage": "starting", "message": "训练进程已启动"})

        assert process.stdout is not None
        for line in process.stdout:
            text = line.strip()
            if not text:
                continue
            try:
                event = json.loads(text)
            except json.JSONDecodeError:
                self._apply_event(job_id, {"message": text[-500:]})
                continue
            if isinstance(event, dict):
                self._apply_event(job_id, event)

        code = process.wait()
        with self._lock:
            self._processes.pop(job_id, None)
            current = self._jobs.get(job_id, {})
            cancelling = current.get("state") == "cancelling"
            interrupted = current.get("state") == "interrupted"
        if interrupted:
            return
        if cancelling:
            self._apply_event(job_id, {"state": "cancelled", "stage": "cancelled", "message": "训练已取消"})
            return
        if code != 0:
            self._apply_event(
                job_id,
                {"state": "failed", "stage": "failed", "error": f"训练进程退出码 {code}", "message": "训练失败"},
            )
            return

        self._apply_event(job_id, {"state": "completed", "stage": "completed", "progress": 1.0, "message": "训练完成"})
        if self._on_complete is not None:
            with self._lock:
                completed = dict(self._jobs.get(job_id, {}))
            try:
                self._on_complete(completed)
            except Exception as error:
                self._apply_event(job_id, {"message": f"训练完成，但自动注册失败: {error}"})
