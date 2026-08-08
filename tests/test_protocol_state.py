import hashlib
import struct
import tempfile
import unittest
from pathlib import Path

from audio_protocol import (
    FLAG_DISCONTINUITY,
    INPUT_HEADER,
    OUTPUT_HEADER,
    build_audio_output_frame,
    parse_audio_input_frame,
)
from file_transfer import UploadManager
from model_registry import ModelRegistry


class AudioProtocolTests(unittest.TestCase):
    def test_v2_input_and_output_headers(self):
        payload = struct.pack(">4f", 0.0, 0.25, -0.5, 1.0)
        frame = INPUT_HEADER.pack(b"RVCA", 2, FLAG_DISCONTINUITY, 32, 7, 9, 16000, 123456) + payload
        parsed = parse_audio_input_frame(frame)
        self.assertIsNotNone(parsed)
        self.assertEqual(parsed.session_id, 7)
        self.assertEqual(parsed.sequence, 9)
        self.assertEqual(parsed.timestamp_ns, 123456)
        self.assertEqual(parsed.payload, payload)

        output = build_audio_output_frame(
            session_id=7,
            sequence=3,
            sample_rate=16000,
            timestamp_ns=222,
            proc_ms=12,
            input_queue_ms=4,
            output_queue_ms=2,
            flags=FLAG_DISCONTINUITY,
            payload=payload,
        )
        header = OUTPUT_HEADER.unpack_from(output)
        self.assertEqual(header[:4], (b"RVCO", 2, FLAG_DISCONTINUITY, 40))
        self.assertEqual(output[40:], payload)


class FileAndRegistryTests(unittest.TestCase):
    def test_upload_collision_and_registry_cleanup(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            uploads = UploadManager(root)
            data1 = b"first"
            data2 = b"second"
            m1 = uploads.init_upload(name="voice.pth", size=len(data1), sha256=hashlib.sha256(data1).hexdigest())
            uploads.write_chunk_sync(upload_id=m1.upload_id, offset=0, payload=data1)
            _, name1 = uploads.finish_sync(upload_id=m1.upload_id)
            self.assertEqual(name1, "voice.pth")

            m2 = uploads.init_upload(name="voice.pth", size=len(data2), sha256=hashlib.sha256(data2).hexdigest())
            uploads.write_chunk_sync(upload_id=m2.upload_id, offset=0, payload=data2)
            _, name2 = uploads.finish_sync(upload_id=m2.upload_id)
            self.assertNotEqual(name1, name2)
            self.assertTrue((uploads.files_dir / name2).exists())

            index_name = "voice.index"
            (uploads.files_dir / index_name).write_bytes(b"idx")
            registry = ModelRegistry(root)
            voice = registry.add_voice_model(
                name="Voice", pth=name1, index=index_name, files_dir=uploads.files_dir
            )
            self.assertEqual(len(voice["models"]), 1)
            registry.remove_file_references(filename=index_name)
            self.assertEqual(registry.list_voice_models()["models"][0]["index"], "")
            registry.remove_file_references(filename=name1)
            self.assertEqual(registry.list_voice_models()["models"], [])

    def test_rejects_traversal(self):
        with tempfile.TemporaryDirectory() as tmp:
            uploads = UploadManager(Path(tmp))
            # basename normalization keeps writes inside files_dir.
            meta = uploads.init_upload(name="../safe.pth", size=1, sha256=hashlib.sha256(b"x").hexdigest())
            self.assertEqual(meta.name, "safe.pth")


if __name__ == "__main__":
    unittest.main()
