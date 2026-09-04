import unittest

import torch

from rvc_core import DEFAULT_INFERENCE_CONFIG, RVCCore


EXPECTED_INFERENCE_DEFAULTS = {
    "block_time": 0.25,
    "crossfade_length": 0.04,
    "extra_time": 2.0,
    "passthrough": False,
    "f0_up_key": 0,
    "formant_shift": 0.0,
    "f0method": "rmvpe",
    "index_rate": 0.5,
    "speaker_id": 0,
    "silence_db_threshold": -70.0,
    "silence_gate_atten": 0.0,
    "input_noise_reduce": False,
    "output_noise_reduce": False,
    "noise_reduce_prop_decrease": 0.9,
    "rms_mix_rate": 0.8,
}


class InferenceDefaultTests(unittest.TestCase):
    def test_declared_defaults_match_client_contract(self):
        self.assertEqual(DEFAULT_INFERENCE_CONFIG, EXPECTED_INFERENCE_DEFAULTS)

    def test_core_applies_declared_defaults(self):
        core = RVCCore({}, device=torch.device("cpu"))
        try:
            self.assertAlmostEqual(core.block_frame / core.sr, 0.25)
            self.assertAlmostEqual(core.crossfade_frame / core.sr, 0.04)
            self.assertAlmostEqual(core.extra_frame / core.sr, 2.0)
            self.assertFalse(core.passthrough)
            self.assertEqual(core.f0_up_key, 0)
            self.assertEqual(core.formant_shift, 0.0)
            self.assertEqual(core.f0_method, "rmvpe")
            self.assertEqual(core.index_rate, 0.5)
            self.assertEqual(core.speaker_id, 0)
            self.assertEqual(core.silence_db_threshold, -70.0)
            self.assertEqual(core.silence_gate_atten, 0.0)
            self.assertFalse(core.input_noise_reduce)
            self.assertFalse(core.output_noise_reduce)
            self.assertEqual(core.noise_reduce_prop_decrease, 0.9)
            self.assertEqual(core.rms_mix_rate, 0.8)
        finally:
            core.close()


if __name__ == "__main__":
    unittest.main()
