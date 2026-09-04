import unittest

import torch

from rvc_core import RVCCore


class _FakeInferer:
    def __init__(self):
        self.prepare_count = 0

    def prepare(self, f0method):
        self.prepare_count += 1
        return {"method": f0method}


class RuntimeWarmupTests(unittest.TestCase):
    def test_warmup_runs_one_exact_live_block_then_resets_state(self):
        core = object.__new__(RVCCore)
        core._inferer = _FakeInferer()
        core.device = torch.device("cpu")
        core.passthrough = False
        core.model_path = "voice.pth"
        core.f0_method = "rmvpe"
        core.block_frame = 4_000
        core.bytes_per_sample = 4
        process_calls = []
        reset_calls = []
        core.process_frame = lambda payload, timestamp: process_calls.append((payload, timestamp))
        core.reset_stream_state = lambda: reset_calls.append(True)

        info = core.warmup()

        self.assertEqual(info, {"method": "rmvpe"})
        self.assertEqual(core._inferer.prepare_count, 1)
        self.assertEqual(len(process_calls), 1)
        self.assertEqual(len(process_calls[0][0]), 4_000 * 4)
        self.assertEqual(process_calls[0][1], 0)
        self.assertEqual(reset_calls, [True])


if __name__ == "__main__":
    unittest.main()
