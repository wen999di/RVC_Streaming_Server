import threading
import tempfile
import unittest
from pathlib import Path

import torch

from rvc_infer import LoadedModelInfo, RealtimeRVCInferer
from server import AudioProcessor


class InferConfigurationTests(unittest.TestCase):
    def test_index_rate_change_keeps_loaded_index(self):
        with tempfile.TemporaryDirectory() as tmp:
            index_path = Path(tmp) / "voice.index"
            index_path.write_bytes(b"index")
            inferer = RealtimeRVCInferer(device=torch.device("cpu"))
            try:
                inferer.configure(
                    model_path="",
                    index_path=str(index_path),
                    index_rate=0.5,
                )
                sentinel = object()
                inferer._faiss_index = sentinel
                inferer._faiss_big_npy = sentinel
                inferer._faiss_resource = sentinel

                changed = inferer.configure(
                    model_path="",
                    index_path=str(index_path),
                    index_rate=0.75,
                )

                self.assertFalse(changed)
                self.assertIs(inferer._faiss_index, sentinel)
                self.assertIs(inferer._faiss_big_npy, sentinel)
                self.assertIs(inferer._faiss_resource, sentinel)
            finally:
                inferer.close()

    def test_prepare_loads_resources_in_dependency_order(self):
        inferer = RealtimeRVCInferer(device=torch.device("cpu"))
        inferer._model_path = "voice.pth"
        calls = []
        info = LoadedModelInfo(tgt_sr=40000, if_f0=1, version="v2")
        inferer._ensure_hubert_loaded = lambda: calls.append("hubert")

        def load_voice():
            calls.append("voice")
            inferer._info = info

        inferer._ensure_active_model_loaded = load_voice
        inferer._ensure_index_loaded = lambda: calls.append("index")
        inferer._ensure_f0_model_loaded = lambda method: calls.append(f"f0:{method}")

        result = inferer._prepare_locked("rmvpe")

        self.assertEqual(result, info)
        self.assertEqual(calls, ["hubert", "voice", "index", "f0:rmvpe"])


class TimestampTests(unittest.TestCase):
    def test_unknown_timestamp_stays_zero_across_packet_chunks(self):
        class FakeCore:
            block_frame = 2
            bytes_per_sample = 4
            ns_per_sample = 100

            def __init__(self):
                self.timestamps = []

            def process_frame(self, chunk, timestamp_ns):
                self.timestamps.append(timestamp_ns)
                return chunk, timestamp_ns

        processor = AudioProcessor.__new__(AudioProcessor)
        processor._lock = threading.RLock()
        processor.core = FakeCore()

        processor.process_packet(bytes(16), ts_start_ns=0)
        self.assertEqual(processor.core.timestamps, [0, 0])


if __name__ == "__main__":
    unittest.main()
