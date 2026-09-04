import tempfile
import unittest
import json
from pathlib import Path
from unittest.mock import patch

import torch

from cuda_graph_runtime import run as run_cuda_graph
from training_manager import TrainingManager
from training_worker import model_shape
from torchgate import TorchGate


class TrainingShapeTests(unittest.TestCase):
    def test_supported_model_shapes_are_hop_aligned(self):
        for sample_rate in (32000, 40000, 48000):
            shape = model_shape(sample_rate)
            self.assertEqual(shape["segment_samples"] % shape["hop"], 0)
            self.assertEqual(
                sample_rate,
                shape["hop"] * 100,
            )


class TrainingManagerTests(unittest.TestCase):
    def test_pretrained_pair_is_loaded_only_when_selected(self):
        with tempfile.TemporaryDirectory() as root:
            base = Path(root)
            files = base / "files"
            files.mkdir()
            for name in ("voice.wav", "hubert.pt", "rmvpe.pt", "G.pth", "D.pth"):
                (files / name).write_bytes(b"placeholder")
            manager = TrainingManager(base_dir=base, files_dir=files)
            with patch.object(manager, "_run_job"):
                job = manager.start(
                    {
                        "name": "voice",
                        "files": [{"name": "voice.wav", "speaker": "voice"}],
                        "hubert": "hubert.pt",
                        "rmvpe": "rmvpe.pt",
                        "use_pretrained": True,
                        "pretrained_g": "G.pth",
                        "pretrained_d": "D.pth",
                    }
                )
            config = json.loads(
                (base / "training_jobs" / job["id"] / "job.json").read_text(encoding="utf-8")
            )
            self.assertTrue(config["use_pretrained"])
            self.assertEqual(Path(config["pretrained_g"]).name, "G.pth")
            self.assertEqual(Path(config["pretrained_d"]).name, "D.pth")

    def test_selected_pretrained_pair_requires_both_slots(self):
        with tempfile.TemporaryDirectory() as root:
            base = Path(root)
            files = base / "files"
            files.mkdir()
            for name in ("voice.wav", "hubert.pt", "rmvpe.pt", "G.pth"):
                (files / name).write_bytes(b"placeholder")
            manager = TrainingManager(base_dir=base, files_dir=files)
            with self.assertRaisesRegex(ValueError, "生成器或判别器槽位尚未配置"):
                manager.start(
                    {
                        "name": "voice",
                        "files": [{"name": "voice.wav", "speaker": "voice"}],
                        "hubert": "hubert.pt",
                        "rmvpe": "rmvpe.pt",
                        "use_pretrained": True,
                        "pretrained_g": "G.pth",
                    }
                )

    def test_start_validates_and_persists_a_job(self):
        with tempfile.TemporaryDirectory() as root:
            base = Path(root)
            files = base / "files"
            files.mkdir()
            for name in ("voice.wav", "hubert.pt", "rmvpe.pt"):
                (files / name).write_bytes(b"placeholder")
            manager = TrainingManager(base_dir=base, files_dir=files)
            with patch.object(manager, "_run_job"):
                job = manager.start(
                    {
                        "name": "voice",
                        "files": [{"name": "voice.wav", "speaker": "voice"}],
                        "hubert": "hubert.pt",
                        "rmvpe": "rmvpe.pt",
                        "sample_rate": 40000,
                        "epochs": 2,
                        "batch_size": 1,
                    }
                )
            snapshot = manager.snapshot()
            self.assertEqual(job["state"], "queued")
            self.assertEqual(job["file_count"], 1)
            self.assertEqual(snapshot["active_id"], job["id"])
            self.assertTrue((base / "training_jobs.json").is_file())

    def test_rejects_non_audio_training_file(self):
        with tempfile.TemporaryDirectory() as root:
            base = Path(root)
            files = base / "files"
            files.mkdir()
            for name in ("notes.txt", "hubert.pt", "rmvpe.pt"):
                (files / name).write_bytes(b"placeholder")
            manager = TrainingManager(base_dir=base, files_dir=files)
            with self.assertRaisesRegex(ValueError, "不支持的训练音频类型"):
                manager.start(
                    {
                        "name": "voice",
                        "files": [{"name": "notes.txt", "speaker": "voice"}],
                        "hubert": "hubert.pt",
                        "rmvpe": "rmvpe.pt",
                    }
                )

    def test_accepts_training_audio_in_server_subdirectories(self):
        with tempfile.TemporaryDirectory() as root:
            base = Path(root)
            files = base / "files"
            nested = files / "dataset" / "speaker_a"
            nested.mkdir(parents=True)
            (nested / "voice.wav").write_bytes(b"placeholder")
            for name in ("hubert.pt", "rmvpe.pt"):
                (files / name).write_bytes(b"placeholder")
            manager = TrainingManager(base_dir=base, files_dir=files)
            with patch.object(manager, "_run_job"):
                job = manager.start(
                    {
                        "name": "voice",
                        "files": [{"name": "dataset/speaker_a/voice.wav", "speaker": "speaker_a"}],
                        "hubert": "hubert.pt",
                        "rmvpe": "rmvpe.pt",
                    }
                )
            config_path = base / "training_jobs" / job["id"] / "job.json"
            config = json.loads(config_path.read_text(encoding="utf-8"))
            self.assertEqual(config["audio_files"][0]["name"], "dataset/speaker_a/voice.wav")

    def test_optional_pymss_preprocessing_is_isolated_and_persisted(self):
        with tempfile.TemporaryDirectory() as root:
            base = Path(root)
            files = base / "files"
            files.mkdir()
            for name in ("voice.wav", "hubert.pt", "rmvpe.pt", "separator.ckpt", "separator.yaml"):
                (files / name).write_bytes(b"placeholder")
            manager = TrainingManager(base_dir=base, files_dir=files)
            with (
                patch.object(manager, "_run_job"),
                patch.object(manager, "_resolve_pymss_python", return_value="pymss-python"),
            ):
                job = manager.start(
                    {
                        "name": "voice",
                        "files": [{"name": "voice.wav", "speaker": "voice"}],
                        "hubert": "hubert.pt",
                        "rmvpe": "rmvpe.pt",
                        "pymss_weight": "separator.ckpt",
                        "pymss_config": "separator.yaml",
                        "preprocess": "vocals",
                    }
                )
            config_path = next((base / "training_jobs" / job["id"]).glob("job.json"))
            config = json.loads(config_path.read_text(encoding="utf-8"))
            self.assertEqual(job["preprocess"], "vocals")
            self.assertEqual(config["pymss_model_type"], "bs_roformer")
            self.assertEqual(config["pymss_stem"], "vocals")
            self.assertEqual(config["pymss_python"], "pymss-python")

    def test_pymss_preprocessing_requires_weight_and_yaml(self):
        with tempfile.TemporaryDirectory() as root:
            base = Path(root)
            files = base / "files"
            files.mkdir()
            for name in ("voice.wav", "hubert.pt", "rmvpe.pt"):
                (files / name).write_bytes(b"placeholder")
            manager = TrainingManager(base_dir=base, files_dir=files)
            with self.assertRaisesRegex(ValueError, "缺少训练输入文件"):
                manager.start(
                    {
                        "name": "voice",
                        "files": [{"name": "voice.wav", "speaker": "voice"}],
                        "hubert": "hubert.pt",
                        "rmvpe": "rmvpe.pt",
                        "preprocess": "noreverb",
                    }
                )


class CudaGraphFallbackTests(unittest.TestCase):
    def test_cpu_uses_eager_function(self):
        owner = object()
        value = torch.tensor([2.0])
        output = run_cuda_graph(owner, "cpu", lambda item: item.square(), value)
        self.assertEqual(output.item(), 4.0)


class TorchGateTests(unittest.TestCase):
    def test_stationary_gate_accepts_a_longer_noise_reference(self):
        gate = TorchGate(sr=16000, nonstationary=False, n_fft=400, win_length=400, hop_length=160)
        audio = torch.randn(1, 1600) * 0.01
        reference = torch.randn(1, 4800) * 0.01
        output = gate(audio, reference)
        self.assertEqual(tuple(output.shape), (1600,))
        self.assertIn("window", dict(gate.named_buffers()))


if __name__ == "__main__":
    unittest.main()
