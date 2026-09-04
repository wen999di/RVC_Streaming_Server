from __future__ import annotations

import os
import re
import shutil
import threading
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Iterable
from urllib.parse import unquote, urlparse

from file_transfer import resolve_relative_file, sanitize_relative_path


_REPO_PART = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,127}$")
_PROVIDERS = {"huggingface", "modelscope"}


@dataclass(frozen=True)
class HubFile:
    path: str
    size: int
    oid: str = ""

    def to_dict(self) -> dict:
        return {"path": self.path, "size": self.size, "oid": self.oid}


@dataclass(frozen=True)
class HubRepository:
    provider: str
    repo_id: str
    revision: str
    files: tuple[HubFile, ...]

    @property
    def total_size(self) -> int:
        return sum(max(0, item.size) for item in self.files)


@dataclass(frozen=True)
class HubDownloadResult:
    request_id: str
    provider: str
    repo_id: str
    revision: str
    destination: str
    files: tuple[str, ...]
    total_bytes: int


ProgressCallback = Callable[[dict], None]


class HubDownloadManager:
    """Lists and downloads selected files from supported model repositories.

    Repository paths are always resolved below ``files_dir``. SDK downloads go
    through a cache first; files are then copied to ``.part`` files and atomically
    replaced at the final path so interrupted downloads never look complete.
    """

    def __init__(self, files_dir: Path, cache_dir: Path | None = None) -> None:
        self.files_dir = Path(files_dir).resolve()
        # Keep SDK caches outside the user-visible file browser. Only finalized
        # selected files belong under files_dir.
        self.cache_dir = Path(cache_dir or self.files_dir.parent / ".hub-cache").resolve()
        self.files_dir.mkdir(parents=True, exist_ok=True)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.max_files = max(1, int(os.environ.get("RVC_HUB_MAX_FILES", "20000")))
        self.max_download_bytes = max(
            1,
            int(os.environ.get("RVC_HUB_MAX_DOWNLOAD_BYTES", str(40 * 1024 * 1024 * 1024))),
        )

    @staticmethod
    def normalize_provider(provider: object) -> str:
        value = str(provider or "").strip().lower().replace("_", "").replace("-", "")
        aliases = {
            "hf": "huggingface",
            "huggingface": "huggingface",
            "modelscope": "modelscope",
            "ms": "modelscope",
        }
        normalized = aliases.get(value, "")
        if normalized not in _PROVIDERS:
            raise ValueError("不支持的模型仓库来源")
        return normalized

    @staticmethod
    def normalize_repo_id(provider: str, value: object) -> str:
        raw = unquote(str(value or "").strip()).replace("\\", "/")
        if not raw:
            raise ValueError("请输入模型仓库地址或仓库 ID")

        if "://" in raw:
            parsed = urlparse(raw)
            host = parsed.netloc.lower().split(":", 1)[0]
            allowed_hosts = {
                "huggingface": {"huggingface.co", "www.huggingface.co"},
                "modelscope": {"modelscope.cn", "www.modelscope.cn", "modelscope.ai", "www.modelscope.ai"},
            }[provider]
            if host not in allowed_hosts:
                raise ValueError("仓库地址与所选来源不一致")
            parts = [part for part in parsed.path.split("/") if part]
            if provider == "modelscope" and parts and parts[0].lower() in {"models", "model"}:
                parts = parts[1:]
            if len(parts) < 2:
                raise ValueError("无法从地址中识别仓库 ID")
            raw = "/".join(parts[:2])

        parts = [part for part in raw.strip("/").split("/") if part]
        if len(parts) == 2 and parts[1].lower().endswith(".git"):
            parts[1] = parts[1][:-4]
        if len(parts) != 2 or not all(_REPO_PART.fullmatch(part) for part in parts):
            raise ValueError("仓库 ID 应为 owner/repo")
        return "/".join(parts)

    @staticmethod
    def normalize_revision(value: object) -> str:
        revision = str(value or "").strip()
        if len(revision) > 200 or "\x00" in revision or revision.startswith(("/", "\\")):
            raise ValueError("无效的仓库版本")
        if any(part in {".", ".."} for part in revision.replace("\\", "/").split("/")):
            raise ValueError("无效的仓库版本")
        return revision

    def default_destination(self, provider: str, repo_id: str) -> str:
        return sanitize_relative_path(f"downloads/{provider}/{repo_id}")

    def list_repository(self, *, provider: object, repo: object, revision: object = "") -> HubRepository:
        normalized_provider = self.normalize_provider(provider)
        repo_id = self.normalize_repo_id(normalized_provider, repo)
        normalized_revision = self.normalize_revision(revision)
        if normalized_provider == "huggingface":
            files = self._list_huggingface(repo_id, normalized_revision)
            effective_revision = normalized_revision or "main"
        else:
            files = self._list_modelscope(repo_id, normalized_revision)
            effective_revision = normalized_revision or "master"
        files = tuple(sorted(files, key=lambda item: item.path.casefold()))
        if len(files) > self.max_files:
            raise ValueError(f"仓库文件过多，最多支持 {self.max_files} 个文件")
        return HubRepository(normalized_provider, repo_id, effective_revision, files)

    def download_selected(
        self,
        *,
        request_id: object,
        provider: object,
        repo: object,
        revision: object,
        paths: Iterable[object],
        destination: object = "",
        cancel_event: threading.Event | None = None,
        progress: ProgressCallback | None = None,
    ) -> HubDownloadResult:
        normalized_request_id = str(request_id or "").strip().lower()
        if not re.fullmatch(r"[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}", normalized_request_id):
            normalized_request_id = str(uuid.uuid4())
        repository = self.list_repository(provider=provider, repo=repo, revision=revision)
        requested_paths: list[str] = []
        seen: set[str] = set()
        for value in paths:
            safe_path = sanitize_relative_path(str(value or ""))
            if safe_path not in seen:
                requested_paths.append(safe_path)
                seen.add(safe_path)
        if not requested_paths:
            raise ValueError("请至少选择一个仓库文件")
        if len(requested_paths) > self.max_files:
            raise ValueError(f"一次最多下载 {self.max_files} 个文件")

        available = {item.path: item for item in repository.files}
        missing = [path for path in requested_paths if path not in available]
        if missing:
            raise ValueError(f"仓库中不存在所选文件：{missing[0]}")
        total_bytes = sum(max(0, available[path].size) for path in requested_paths)
        if total_bytes > self.max_download_bytes:
            raise ValueError("所选文件总大小超过服务器下载限制")

        destination_value = str(destination or "").strip()
        destination_safe = (
            sanitize_relative_path(destination_value)
            if destination_value
            else self.default_destination(repository.provider, repository.repo_id)
        )
        _, destination_root = resolve_relative_file(self.files_dir, destination_safe)
        destination_root.mkdir(parents=True, exist_ok=True)

        completed_bytes = 0
        completed_files: list[str] = []
        cancel_event = cancel_event or threading.Event()

        for index, remote_path in enumerate(requested_paths, start=1):
            if cancel_event.is_set():
                raise InterruptedError("下载已取消")
            item = available[remote_path]
            if progress:
                progress({
                    "request_id": normalized_request_id,
                    "path": remote_path,
                    "file_index": index,
                    "file_count": len(requested_paths),
                    "completed_bytes": completed_bytes,
                    "total_bytes": total_bytes,
                    "state": "downloading",
                })

            cached_path = self._download_to_cache(repository, remote_path)
            relative_path, target_path = resolve_relative_file(destination_root, remote_path)
            target_path.parent.mkdir(parents=True, exist_ok=True)
            part_path = target_path.with_name(f".{target_path.name}.{normalized_request_id}.part")
            try:
                self._copy_atomic(cached_path, part_path, target_path, cancel_event)
            finally:
                part_path.unlink(missing_ok=True)

            completed_bytes += max(0, item.size)
            completed_name = sanitize_relative_path(f"{destination_safe}/{relative_path}")
            completed_files.append(completed_name)
            if progress:
                progress({
                    "request_id": normalized_request_id,
                    "path": remote_path,
                    "name": completed_name,
                    "file_index": index,
                    "file_count": len(requested_paths),
                    "completed_bytes": completed_bytes,
                    "total_bytes": total_bytes,
                    "state": "completed",
                })

        return HubDownloadResult(
            request_id=normalized_request_id,
            provider=repository.provider,
            repo_id=repository.repo_id,
            revision=repository.revision,
            destination=destination_safe,
            files=tuple(completed_files),
            total_bytes=total_bytes,
        )

    def _list_huggingface(self, repo_id: str, revision: str) -> list[HubFile]:
        try:
            from huggingface_hub import HfApi, RepoFile
        except ImportError as exc:
            raise RuntimeError("服务器未安装 huggingface-hub") from exc
        token = os.environ.get("HF_TOKEN") or None
        api = HfApi(token=token)
        items = api.list_repo_tree(
            repo_id=repo_id,
            repo_type="model",
            revision=revision or None,
            recursive=True,
            expand=False,
            token=token,
        )
        files: list[HubFile] = []
        for item in items:
            if not isinstance(item, RepoFile):
                continue
            try:
                path = sanitize_relative_path(str(item.path))
            except ValueError:
                # A repository can contain names that Windows cannot create.
                # Keep the rest of the repository browsable instead of failing it all.
                continue
            lfs = getattr(item, "lfs", None)
            oid = str(getattr(lfs, "sha256", "") or getattr(item, "blob_id", "") or "")
            files.append(HubFile(path, max(0, int(getattr(item, "size", 0) or 0)), oid))
        return files

    def _list_modelscope(self, repo_id: str, revision: str) -> list[HubFile]:
        try:
            from modelscope_hub import HubApi
        except ImportError as exc:
            raise RuntimeError("服务器未安装 modelscope-hub") from exc
        token = os.environ.get("MODELSCOPE_API_TOKEN") or os.environ.get("MODELSCOPE_TOKEN") or None
        endpoint = os.environ.get("MODELSCOPE_DOMAIN", "https://modelscope.cn")
        if endpoint and not endpoint.startswith(("http://", "https://")):
            endpoint = f"https://{endpoint}"
        api = HubApi(token=token, endpoint=endpoint)
        raw_files = api.list_repo_files(
            repo_id=repo_id,
            repo_type="model",
            revision=revision or None,
            recursive=True,
        )
        files: list[HubFile] = []
        for item in raw_files:
            if isinstance(item, dict):
                path = item.get("path") or item.get("Path") or item.get("name") or item.get("Name")
                size = item.get("size") or item.get("Size") or 0
                oid = item.get("sha256") or item.get("Sha256") or item.get("blob_id") or ""
                item_type = str(item.get("type") or item.get("Type") or "blob").lower()
            else:
                path = getattr(item, "path", None) or getattr(item, "name", None)
                size = getattr(item, "size", 0) or 0
                oid = getattr(item, "sha256", "") or getattr(item, "blob_id", "") or ""
                item_type = str(getattr(item, "type", "blob") or "blob").lower()
            if not path or item_type in {"tree", "folder", "directory"}:
                continue
            try:
                safe_path = sanitize_relative_path(str(path))
            except ValueError:
                continue
            files.append(HubFile(safe_path, max(0, int(size)), str(oid)))
        return files

    def _download_to_cache(self, repository: HubRepository, remote_path: str) -> Path:
        if repository.provider == "huggingface":
            from huggingface_hub import hf_hub_download

            result = hf_hub_download(
                repo_id=repository.repo_id,
                filename=remote_path,
                repo_type="model",
                revision=repository.revision,
                cache_dir=str(self.cache_dir / "huggingface"),
                token=os.environ.get("HF_TOKEN") or None,
            )
            return Path(result).resolve()

        from modelscope_hub import HubApi

        token = os.environ.get("MODELSCOPE_API_TOKEN") or os.environ.get("MODELSCOPE_TOKEN") or None
        endpoint = os.environ.get("MODELSCOPE_DOMAIN", "https://modelscope.cn")
        if endpoint and not endpoint.startswith(("http://", "https://")):
            endpoint = f"https://{endpoint}"
        api = HubApi(token=token, endpoint=endpoint)
        try:
            result = api.download_file(
                repo_id=repository.repo_id,
                repo_type="model",
                file_path=remote_path,
                revision=repository.revision,
                cache_dir=str(self.cache_dir / "modelscope"),
            )
        except TypeError:
            result = api.download_file(
                repository.repo_id,
                "model",
                remote_path,
                revision=repository.revision,
                cache_dir=str(self.cache_dir / "modelscope"),
            )
        return Path(result).resolve()

    @staticmethod
    def _copy_atomic(source: Path, part: Path, target: Path, cancel_event: threading.Event) -> None:
        with source.open("rb") as src, part.open("wb") as dst:
            while True:
                if cancel_event.is_set():
                    raise InterruptedError("下载已取消")
                chunk = src.read(4 * 1024 * 1024)
                if not chunk:
                    break
                dst.write(chunk)
            dst.flush()
            os.fsync(dst.fileno())
        os.replace(part, target)
