import hashlib
import json
import os
import time
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Tuple
import struct


FILE_MAGIC = b"RVCFILE1"
FILE_CHUNK_TYPE = 1


def _now_s() -> float:
    return time.time()


def sanitize_filename(name: str) -> str:
    if not isinstance(name, str):
        raise ValueError("invalid filename")
    name = name.strip()
    name = os.path.basename(name)
    if not name or name in (".", ".."):
        raise ValueError("invalid filename")
    if any(sep in name for sep in ("/", "\\", "\x00")):
        raise ValueError("invalid filename")
    if len(name) > 255:
        raise ValueError("filename too long")
    return name


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
        self.base_dir = base_dir or Path(__file__).resolve().parent
        self.files_dir = self.base_dir / "files"
        self.partial_dir = self.files_dir / ".partial"
        self.uploads_dir = self.base_dir / "uploads"

        self.files_dir.mkdir(parents=True, exist_ok=True)
        self.partial_dir.mkdir(parents=True, exist_ok=True)
        self.uploads_dir.mkdir(parents=True, exist_ok=True)

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
                    continue
                if meta.received_bytes < 0 or meta.received_bytes > meta.size:
                    continue
                self._uploads[meta.upload_id] = meta
                if meta.key:
                    self._key_to_upload_id[meta.key] = meta.upload_id
            except Exception:
                continue

    def init_upload(self, *, name: str, size: int, sha256: str) -> UploadMeta:
        safe_name = sanitize_filename(name)
        if size <= 0:
            raise ValueError("invalid size")
        sha256 = (sha256 or "").strip().lower()
        if sha256 and len(sha256) != 64:
            raise ValueError("invalid sha256")

        key = sha256 or f"name:{safe_name}|size:{size}"
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

    def get(self, upload_id: str) -> UploadMeta:
        meta = self._uploads.get(upload_id)
        if not meta:
            raise KeyError("unknown upload_id")
        return meta

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

        target = self.files_dir / meta.name
        if target.exists():
            stem = target.stem
            suffix = target.suffix
            disambiguator = time.strftime("%Y%m%d_%H%M%S")
            target = self.files_dir / f"{stem}_{disambiguator}{suffix}"

        os.replace(part, target)

        self._meta_path(upload_id).unlink(missing_ok=True)
        self._uploads.pop(upload_id, None)
        if meta.key and self._key_to_upload_id.get(meta.key) == upload_id:
            self._key_to_upload_id.pop(meta.key, None)
        self._last_meta_flush_s.pop(upload_id, None)
        return meta, target.name

    def abort_sync(self, *, upload_id: str) -> None:
        meta = self._uploads.pop(upload_id, None)
        if meta and meta.key and self._key_to_upload_id.get(meta.key) == upload_id:
            self._key_to_upload_id.pop(meta.key, None)
        self._last_meta_flush_s.pop(upload_id, None)
        self._meta_path(upload_id).unlink(missing_ok=True)
        self._part_path(upload_id).unlink(missing_ok=True)

    def list_files(self) -> list[dict]:
        items: list[dict] = []
        for p in self.files_dir.iterdir():
            if not p.is_file():
                continue
            st = p.stat()
            items.append(
                {
                    "name": p.name,
                    "size": int(st.st_size),
                    "mtime": float(st.st_mtime),
                }
            )
        items.sort(key=lambda x: x["mtime"], reverse=True)
        return items

    def delete_file(self, *, name: str) -> None:
        safe_name = sanitize_filename(name)
        target = self.files_dir / safe_name
        if not target.exists():
            raise FileNotFoundError("file_not_found")
        if not target.is_file():
            raise ValueError("not_a_file")
        target.unlink()

    def rename_file(self, *, old_name: str, new_name: str) -> str:
        old_safe = sanitize_filename(old_name)
        new_safe = sanitize_filename(new_name)
        if old_safe.lower() == new_safe.lower():
            return new_safe

        src = self.files_dir / old_safe
        if not src.exists():
            raise FileNotFoundError("file_not_found")
        if not src.is_file():
            raise ValueError("not_a_file")

        dst = self.files_dir / new_safe
        if dst.exists():
            raise FileExistsError("target_exists")

        os.replace(src, dst)
        return dst.name

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
