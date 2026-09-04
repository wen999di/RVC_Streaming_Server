import unittest
import tempfile
from contextlib import nullcontext
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import torch
import server

from rvc_infer import RealtimeRVCInferer
from server import _required_base_model_slot_error


class StrictBaseModelSlotTests(unittest.TestCase):
    def test_hubert_requires_an_explicitly_configured_slot(self):
        inferer = RealtimeRVCInferer(device=torch.device("cpu"))
        inferer._hubert = object()

        with patch("rvc_infer._load_hubert") as load_hubert:
            with self.assertRaisesRegex(RuntimeError, "未配置 HuBERT Base 模型槽位"):
                inferer._ensure_hubert_loaded()

        load_hubert.assert_not_called()

    def test_rmvpe_requires_an_explicitly_configured_slot(self):
        inferer = RealtimeRVCInferer(device=torch.device("cpu"))
        inferer._rmvpe = object()

        with patch("rvc_infer._load_rmvpe") as load_rmvpe:
            with self.assertRaisesRegex(RuntimeError, "未配置 RMVPE 模型槽位"):
                inferer._ensure_rmvpe_loaded()

        load_rmvpe.assert_not_called()

    def test_server_reports_all_missing_required_slots_before_loading(self):
        error = _required_base_model_slot_error(
            {"model_path": "voice.pth", "f0method": "rmvpe", "hubert_path": "", "rmvpe_path": ""}
        )

        self.assertEqual(error, "未配置 HuBERT Base 和 RMVPE 模型槽位")

    def test_fcpe_does_not_require_rmvpe_slot(self):
        error = _required_base_model_slot_error(
            {"model_path": "voice.pth", "f0method": "fcpe", "hubert_path": "hubert.pt", "rmvpe_path": ""}
        )

        self.assertIsNone(error)

    def test_preload_skips_default_named_files_when_slots_are_empty(self):
        with tempfile.TemporaryDirectory() as tmp:
            files_dir = Path(tmp)
            (files_dir / "hubert_base.pt").write_bytes(b"unused")
            (files_dir / "rmvpe.pt").write_bytes(b"unused")
            registry = SimpleNamespace(
                list_slots=lambda: {
                    "hubert_base": {"active": ""},
                    "rmvpe": {"active": ""},
                }
            )
            uploads = SimpleNamespace(files_dir=files_dir)

            with (
                patch.object(server, "model_registry", registry),
                patch.object(server, "upload_manager", uploads),
                patch("rvc_infer._load_hubert") as load_hubert,
                patch("rvc_infer._load_rmvpe") as load_rmvpe,
            ):
                server._preload_base_models()

            load_hubert.assert_not_called()
            load_rmvpe.assert_not_called()

    def test_preload_loads_both_explicitly_configured_base_model_slots(self):
        with tempfile.TemporaryDirectory() as tmp:
            files_dir = Path(tmp)
            hubert_path = files_dir / "custom-hubert.pt"
            rmvpe_path = files_dir / "custom-rmvpe.pt"
            hubert_path.write_bytes(b"hubert")
            rmvpe_path.write_bytes(b"rmvpe")
            registry = SimpleNamespace(
                list_slots=lambda: {
                    "hubert_base": {"active": hubert_path.name},
                    "rmvpe": {"active": rmvpe_path.name},
                }
            )
            uploads = SimpleNamespace(files_dir=files_dir)

            with (
                patch.object(server, "model_registry", registry),
                patch.object(server, "upload_manager", uploads),
                patch.object(torch.cuda, "is_available", return_value=False),
                patch("rvc_infer._device_infer_lock", return_value=nullcontext()),
                patch("rvc_infer._load_hubert") as load_hubert,
                patch("rvc_infer._load_rmvpe") as load_rmvpe,
            ):
                server._preload_base_models()

            load_hubert.assert_called_once()
            load_rmvpe.assert_called_once()
            self.assertEqual(load_hubert.call_args.args[2], str(hubert_path))
            self.assertEqual(load_rmvpe.call_args.args[2], str(rmvpe_path))

    def test_preload_scheduler_coalesces_changes_during_a_running_preload(self):
        class FakeLoop:
            def is_closed(self):
                return False

        work = []

        class FakeThread:
            def __init__(self, *, target, name, daemon):
                self.target = target
                self.name = name
                self.daemon = daemon

            def start(self):
                work.append(self.target)

        loop = FakeLoop()

        def request_refresh_during_first_preload():
            if preload.call_count == 1:
                self.assertFalse(server._schedule_base_model_preload(loop))

        with (
            patch.object(server, "_base_model_preload_running", False),
            patch.object(server, "_base_model_preload_pending", False),
            patch.object(server, "_preload_base_models", side_effect=request_refresh_during_first_preload) as preload,
            patch.object(server.threading, "Thread", FakeThread),
        ):
            self.assertTrue(server._schedule_base_model_preload(loop))
            self.assertEqual(len(work), 1)
            work[0]()

            self.assertEqual(preload.call_count, 2)
            self.assertFalse(server._base_model_preload_running)



if __name__ == "__main__":
    unittest.main()
