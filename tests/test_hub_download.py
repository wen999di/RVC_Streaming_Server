import tempfile
import threading
import unittest
from pathlib import Path

from hub_download import HubDownloadManager, HubFile, HubRepository


class FakeHubDownloadManager(HubDownloadManager):
    def list_repository(self, *, provider, repo, revision=""):
        normalized_provider = self.normalize_provider(provider)
        repo_id = self.normalize_repo_id(normalized_provider, repo)
        return HubRepository(
            normalized_provider,
            repo_id,
            str(revision or "main"),
            (
                HubFile("weights/model.pth", 5, "a"),
                HubFile("config/model.yaml", 6, "b"),
            ),
        )

    def _download_to_cache(self, repository, remote_path):
        target = self.cache_dir / repository.provider / repository.repo_id / remote_path
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(b"model" if remote_path.endswith(".pth") else b"config")
        return target


class HubDownloadTests(unittest.TestCase):
    def test_repository_urls_are_normalized(self):
        self.assertEqual(
            HubDownloadManager.normalize_repo_id(
                "huggingface", "https://huggingface.co/acme/rvc-model/tree/main"
            ),
            "acme/rvc-model",
        )
        self.assertEqual(
            HubDownloadManager.normalize_repo_id(
                "modelscope", "https://modelscope.cn/models/acme/rvc-model"
            ),
            "acme/rvc-model",
        )

    def test_selected_files_keep_repository_paths_without_random_suffixes(self):
        with tempfile.TemporaryDirectory() as tmp:
            manager = FakeHubDownloadManager(Path(tmp) / "files")
            result = manager.download_selected(
                request_id="11111111-1111-1111-1111-111111111111",
                provider="hf",
                repo="acme/rvc-model",
                revision="main",
                paths=["weights/model.pth", "config/model.yaml"],
                destination="downloads/huggingface/acme/rvc-model",
            )
            self.assertEqual(
                list(result.files),
                [
                    "downloads/huggingface/acme/rvc-model/weights/model.pth",
                    "downloads/huggingface/acme/rvc-model/config/model.yaml",
                ],
            )
            self.assertEqual(
                (manager.files_dir / result.files[0]).read_bytes(),
                b"model",
            )
            self.assertFalse(any(manager.files_dir.rglob("*_*.*")))

    def test_cancelled_download_does_not_create_partial_files(self):
        with tempfile.TemporaryDirectory() as tmp:
            manager = FakeHubDownloadManager(Path(tmp) / "files")
            cancelled = threading.Event()
            cancelled.set()
            with self.assertRaises(InterruptedError):
                manager.download_selected(
                    request_id="22222222-2222-2222-2222-222222222222",
                    provider="modelscope",
                    repo="acme/rvc-model",
                    revision="master",
                    paths=["weights/model.pth"],
                    cancel_event=cancelled,
                )
            self.assertFalse(any(manager.files_dir.rglob("*.part")))

    def test_destination_cannot_escape_server_files(self):
        with tempfile.TemporaryDirectory() as tmp:
            manager = FakeHubDownloadManager(Path(tmp) / "files")
            with self.assertRaises(ValueError):
                manager.download_selected(
                    request_id="33333333-3333-3333-3333-333333333333",
                    provider="hf",
                    repo="acme/rvc-model",
                    revision="main",
                    paths=["weights/model.pth"],
                    destination="../outside",
                )


if __name__ == "__main__":
    unittest.main()
