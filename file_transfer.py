import hashlib
import json
import os
import time
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping, Optional, Tuple
import struct
import threading
import functools


def _synchronized(method):
    @functools.wraps(method)
    def wrapper(self, *args, **kwargs):
        with self._lock:
            return method(self, *args, **kwargs)
    return wrapper


FILE_MAGIC = b"RVCFILE1"
FILE_CHUNK_TYPE = 1
TRAINING_AUDIO_SUFFIXES = {".wav", ".flac", ".mp3", ".m4a", ".ogg", ".opus"}


def _now_s() -> float:
    return time.time()


def sanitize_relative_path(name: str) -> str:
    if not isinstance(name, str):
        raise ValueError("invalid filename")
    raw = name.strip().replace("\\", "/")
    if not raw or raw.startswith("/") or len(raw) > 2048 or "\x00" in raw:
        raise ValueError("invalid filename")
    parts = raw.split("/")
    forbidden = set('<>:"|?*')
    if any(
        not part
        or part in (".", "..")
        or len(part) > 255
        or part[-1] in (" ", ".")
        or any(ord(ch) < 32 or ch in forbidden for ch in part)
        for part in parts
    ):
        raise ValueError("invalid filename")
    return "/".join(parts)


def resolve_relative_file(root: Path, name: str) -> tuple[str, Path]:
    safe_name = sanitize_relative_path(name)
    root = root.resolve()
    target = (root / Path(*safe_name.split("/"))).resolve()
    if root not in target.parents:
        raise ValueError("invalid filename")
    return safe_name, target


def _sanitize_dataset_segment(value: object, fallback: str, max_length: int = 80) -> str:
    raw = str(value or "").strip()
    forbidden = set('<>:"/\\|?*')
    cleaned = "".join(
        "_" if ord(ch) < 32 or ch in forbidden else ch
        for ch in raw
    ).strip(" .")
    cleaned = cleaned[:max_length].rstrip(" .") or fallback
    reserved = {
        "CON", "PRN", "AUX", "NUL",
        *(f"COM{i}" for i in range(1, 10)),
        *(f"LPT{i}" for i in range(1, 10)),
    }
    if cleaned.upper() in reserved:
        cleaned = f"_{cleaned}"
    return cleaned


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def parse_file_chunk_frame(frame: bytes) -> Optional[Tuple[uuid.UUID, int, bytes]]:
    if len(frame) < 8:
        return None
    if frame[:8] != FILE_MAGIC:
        return None
    if len(frame) < 8 + 1 + 16 + 8 + 4:
        raise ValueError("file frame too short")
    msg_type = frame[8]
    if msg_type != FILE_CHUNK_TYPE:
        raise ValueError("unsupported file frame type")
    upload_id_bytes = frame[9:25]
    upload_id = uuid.UUID(bytes=upload_id_bytes)
    offset = struct.unpack(">Q", frame[25:33])[0]
    length = struct.unpack(">I", frame[33:37])[0]
    payload = frame[37:]
    if len(payload) != length:
        raise ValueError("file frame length mismatch")
    return upload_id, offset, payload


@dataclass
class UploadMeta:
    upload_id: str
    key: str
    name: str
    size: int
    sha256: str
    received_bytes: int
    created_at: float
    updated_at: float

    def to_dict(self) -> dict:
        return {
            "upload_id": self.upload_id,
            "key": self.key,
            "name": self.name,
            "size": self.size,
            "sha256": self.sha256,
            "received_bytes": self.received_bytes,
            "created_at": self.created_at,
            "updated_at": self.updated_at,
        }

    @staticmethod
    def from_dict(d: dict) -> "UploadMeta":
        return UploadMeta(
            upload_id=str(d["upload_id"]),
            key=str(d.get("key") or d.get("sha256") or ""),
            name=str(d["name"]),
            size=int(d["size"]),
            sha256=str(d.get("sha256") or ""),
            received_bytes=int(d.get("received_bytes") or 0),
            created_at=float(d.get("created_at") or _now_s()),
            updated_at=float(d.get("updated_at") or _now_s()),
        )


class UploadManager:
    def __init__(self, base_dir: Optional[Path] = None) -> None:
        self.base_dir = (base_dir or Path(__file__).resolve().parent).resolve()
        self.files_dir = self.base_dir / "files"
        self.partial_dir = self.files_dir / ".partial"
        self.uploads_dir = self.base_dir / "uploads"

        self.files_dir.mkdir(parents=True, exist_ok=True)
        self.partial_dir.mkdir(parents=True, exist_ok=True)
        self.uploads_dir.mkdir(parents=True, exist_ok=True)

        self.max_upload_bytes = max(1, int(os.environ.get("RVC_MAX_UPLOAD_BYTES", str(2 * 1024 * 1024 * 1024))))
        self.max_active_uploads = max(1, int(os.environ.get("RVC_MAX_ACTIVE_UPLOADS", "8")))
        self.max_reserved_upload_bytes = max(
            self.max_upload_bytes,
            int(os.environ.get("RVC_MAX_RESERVED_UPLOAD_BYTES", str(4 * 1024 * 1024 * 1024))),
        )
        self.stale_upload_seconds = max(300, int(os.environ.get("RVC_STALE_UPLOAD_SECONDS", "86400")))
        self._lock = threading.RLock()
        self._uploads: dict[str, UploadMeta] = {}
        self._key_to_upload_id: dict[str, str] = {}
        self._last_meta_flush_s: dict[str, float] = {}

        self._load_existing()

    def _meta_path(self, upload_id: str) -> Path:
        return self.uploads_dir / f"{upload_id}.json"

    def _part_path(self, upload_id: str) -> Path:
        return self.partial_dir / f"{upload_id}.part"

    def _load_existing(self) -> None:
        for p in self.uploads_dir.glob("*.json"):
            try:
                with open(p, "r", encoding="utf-8") as f:
                    meta = UploadMeta.from_dict(json.load(f))
                part = self._part_path(meta.upload_id)
                if not part.exists():
                    p.unlink(missing_ok=True)
                    continue
                if _now_s() - float(meta.updated_at) > self.stale_upload_seconds:
                    p.unlink(missing_ok=True)
                    part.unlink(missing_ok=True)
                    continue
                if meta.received_bytes < 0 or meta.received_bytes > meta.size:
                    p.unlink(missing_ok=True)
                    part.unlink(missing_ok=True)
                    continue
                self._uploads[meta.upload_id] = meta
                if meta.key:
                    self._key_to_upload_id[meta.key] = meta.upload_id
            except Exception:
                continue

    @_synchronized
    def init_upload(self, *, name: str, size: int, sha256: str) -> UploadMeta:
        safe_name = sanitize_relative_path(name)
        if size < 0:
            raise ValueError("invalid size")
        if size > self.max_upload_bytes:
            raise ValueError("file_too_large")
        sha256 = (sha256 or "").strip().lower()
        if sha256 and (len(sha256) != 64 or any(ch not in "0123456789abcdef" for ch in sha256)):
            raise ValueError("invalid sha256")

        key = f"name:{safe_name.lower()}|size:{size}|sha256:{sha256}"
        existing_id = self._key_to_upload_id.get(key)
        if existing_id:
            existing = self._uploads.get(existing_id)
            if existing and existing.size == size:
                part = self._part_path(existing.upload_id)
                if part.exists():
                    actual_size = part.stat().st_size
                    if 0 <= actual_size <= size:
                        if existing.received_bytes == 0:
                            existing.name = safe_name
                        existing.received_bytes = int(actual_size)
                        existing.updated_at = _now_s()
                        self._uploads[existing.upload_id] = existing
                        self._flush_meta(existing, force=True)
                        return existing

        if len(self._uploads) >= self.max_active_uploads:
            raise ValueError("too_many_active_uploads")
        reserved = sum(max(0, int(item.size)) for item in self._uploads.values())
        if reserved + int(size) > self.max_reserved_upload_bytes:
            raise ValueError("upload_reservation_limit")

        upload_id = str(uuid.uuid4())
        meta = UploadMeta(
            upload_id=upload_id,
            key=key,
            name=safe_name,
            size=int(size),
            sha256=sha256,
            received_bytes=0,
            created_at=_now_s(),
            updated_at=_now_s(),
        )
        part = self._part_path(upload_id)
        with open(part, "wb") as f:
            f.truncate(0)
        self._uploads[upload_id] = meta
        self._key_to_upload_id[key] = upload_id
        self._flush_meta(meta, force=True)
        return meta

    @_synchronized
    def get(self, upload_id: str) -> UploadMeta:
        meta = self._uploads.get(upload_id)
        if not meta:
            raise KeyError("unknown upload_id")
        return meta

    @_synchronized
    def write_chunk_sync(self, *, upload_id: str, offset: int, payload: bytes) -> UploadMeta:
        meta = self.get(upload_id)
        if offset != meta.received_bytes:
            raise ValueError(f"offset_mismatch:{meta.received_bytes}")
        if not payload:
            return meta
        if meta.received_bytes + len(payload) > meta.size:
            raise ValueError("chunk_out_of_range")

        part = self._part_path(upload_id)
        if not part.exists():
            raise FileNotFoundError("partial_not_found")

        with open(part, "r+b") as f:
            f.seek(offset)
            f.write(payload)

        meta.received_bytes += len(payload)
        meta.updated_at = _now_s()
        self._uploads[upload_id] = meta
        self._flush_meta(meta, force=False)
        return meta

    @_synchronized
    def finish_sync(self, *, upload_id: str) -> Tuple[UploadMeta, str]:
        meta = self.get(upload_id)
        if meta.received_bytes != meta.size:
            raise ValueError("incomplete_upload")

        part = self._part_path(upload_id)
        if not part.exists():
            raise FileNotFoundError("partial_not_found")

        if meta.sha256:
            actual = sha256_file(part)
            if actual.lower() != meta.sha256.lower():
                raise ValueError("sha256_mismatch")

        _, target = resolve_relative_file(self.files_dir, meta.name)
        target.parent.mkdir(parents=True, exist_ok=True)
        if target.exists() and not target.is_file():
            raise ValueError("target_not_a_file")

        os.replace(part, target)

        self._meta_path(upload_id).unlink(missing_ok=True)
        self._uploads.pop(upload_id, None)
        if meta.key and self._key_to_upload_id.get(meta.key) == upload_id:
            self._key_to_upload_id.pop(meta.key, None)
        self._last_meta_flush_s.pop(upload_id, None)
        return meta, target.relative_to(self.files_dir).as_posix()

    @_synchronized
    def abort_sync(self, *, upload_id: str) -> None:
        meta = self._uploads.pop(upload_id, None)
        if meta and meta.key and self._key_to_upload_id.get(meta.key) == upload_id:
            self._key_to_upload_id.pop(meta.key, None)
        self._last_meta_flush_s.pop(upload_id, None)
        self._meta_path(upload_id).unlink(missing_ok=True)
        self._part_path(upload_id).unlink(missing_ok=True)

    @_synchronized
    def list_files(self) -> list[dict]:
        items: list[dict] = []
        for p in self.files_dir.rglob("*"):
            if not p.is_file():
                continue
            if self.partial_dir == p.parent or self.partial_dir in p.parents:
                continue
            st = p.stat()
            items.append(
                {
                    "name": p.relative_to(self.files_dir).as_posix(),
                    "size": int(st.st_size),
                    "mtime": float(st.st_mtime),
                }
            )
        items.sort(key=lambda x: x["mtime"], reverse=True)
        return items

    @_synchronized
    def delete_file(self, *, name: str) -> None:
        _, target = resolve_relative_file(self.files_dir, name)
        if not target.exists():
            raise FileNotFoundError("file_not_found")
        if not target.is_file():
            raise ValueError("not_a_file")
        target.unlink()
        parent = target.parent
        while parent != self.files_dir and parent.is_dir() and not any(parent.iterdir()):
            parent.rmdir()
            parent = parent.parent

    @_synchronized
    def rename_file(self, *, old_name: str, new_name: str) -> str:
        old_safe, src = resolve_relative_file(self.files_dir, old_name)
        new_safe, dst = resolve_relative_file(self.files_dir, new_name)
        if old_safe.lower() == new_safe.lower():
            return new_safe
        if not src.exists():
            raise FileNotFoundError("file_not_found")
        if not src.is_file():
            raise ValueError("not_a_file")

        if dst.exists():
            raise FileExistsError("target_exists")

        dst.parent.mkdir(parents=True, exist_ok=True)
        os.replace(src, dst)
        parent = src.parent
        while parent != self.files_dir and parent.is_dir() and not any(parent.iterdir()):
            parent.rmdir()
            parent = parent.parent
        return dst.relative_to(self.files_dir).as_posix()

    @_synchronized
    def organize_training_files(self, *, model_name: object, files: object) -> dict:
        if not isinstance(files, list) or not files:
            raise ValueError("没有可整理的训练音频")
        if len(files) > 20000:
            raise ValueError("训练音频文件数量不能超过 20000")

        model_segment = _sanitize_dataset_segment(model_name, "my model")
        entries: list[tuple[str, Path, str, str]] = []
        seen_sources: set[str] = set()
        for raw in files:
            item = raw if isinstance(raw, Mapping) else {"name": raw}
            old_safe, source = resolve_relative_file(self.files_dir, str(item.get("name") or ""))
            if old_safe.lower() in seen_sources:
                raise ValueError(f"训练音频重复: {old_safe}")
            if not source.is_file():
                raise FileNotFoundError(old_safe)
            suffix = source.suffix.lower()
            if suffix not in TRAINING_AUDIO_SUFFIXES:
                raise ValueError(f"不支持的训练音频类型: {old_safe}")
            speaker = _sanitize_dataset_segment(item.get("speaker"), "默认说话人")
            stem = _sanitize_dataset_segment(source.stem, "audio", max_length=180)
            entries.append((old_safe, source, speaker, f"{stem}{suffix}"))
            seen_sources.add(old_safe.lower())

        plan: list[tuple[str, Path, str, Path]] = []
        planned_targets: set[Path] = set()
        for old_safe, source, speaker, filename in entries:
            base_stem = Path(filename).stem
            suffix = Path(filename).suffix
            attempt = 1
            while True:
                candidate_name = filename if attempt == 1 else f"{base_stem}_{attempt}{suffix}"
                new_safe, target = resolve_relative_file(
                    self.files_dir,
                    f"{model_segment}/dataset/{speaker}/{candidate_name}",
                )
                occupied = target in planned_targets or (target.exists() and target != source)
                if not occupied:
                    break
                attempt += 1
            planned_targets.add(target)
            plan.append((old_safe, source, new_safe, target))

        moved: list[tuple[Path, Path]] = []
        try:
            for _, source, _, target in plan:
                if source == target:
                    continue
                target.parent.mkdir(parents=True, exist_ok=True)
                os.replace(source, target)
                moved.append((source, target))
        except Exception:
            for source, target in reversed(moved):
                try:
                    source.parent.mkdir(parents=True, exist_ok=True)
                    os.replace(target, source)
                except Exception:
                    pass
            raise

        for _, source, _, _ in plan:
            parent = source.parent
            while parent != self.files_dir and parent.is_dir() and not any(parent.iterdir()):
                parent.rmdir()
                parent = parent.parent

        return {
            "model": model_segment,
            "dataset_root": f"{model_segment}/dataset",
            "files": [
                {"old_name": old_safe, "new_name": new_safe}
                for old_safe, _, new_safe, _ in plan
            ],
        }

    def _flush_meta(self, meta: UploadMeta, *, force: bool) -> None:
        now = _now_s()
        last = self._last_meta_flush_s.get(meta.upload_id, 0.0)
        if not force and now - last < 0.5:
            return
        tmp = self._meta_path(meta.upload_id).with_suffix(".json.tmp")
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(meta.to_dict(), f, ensure_ascii=False)
        os.replace(tmp, self._meta_path(meta.upload_id))
        self._last_meta_flush_s[meta.upload_id] = now
