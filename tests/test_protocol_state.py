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
    def test_training_pretrained_slots_are_exposed_and_accept_weights(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            files = root / "files"
            weights = files / "weights"
            weights.mkdir(parents=True)
            (weights / "G.pth").write_bytes(b"generator")
            (files / "D.pth").write_bytes(b"discriminator")
            registry = ModelRegistry(root)

            generator = registry.add_to_slot(
                slot="pretrained_g", filename="weights/G.pth", files_dir=files
            )
            discriminator = registry.add_to_slot(
                slot="pretrained_d", filename="D.pth", files_dir=files
            )

            self.assertEqual(generator["active"], "weights/G.pth")
            self.assertEqual(discriminator["active"], "D.pth")
            self.assertEqual(generator["allowed_ext"], [".pth", ".pt"])

    def test_zero_byte_file_upload_finishes_normally(self):
        with tempfile.TemporaryDirectory() as tmp:
            uploads = UploadManager(Path(tmp))
            empty_hash = hashlib.sha256(b"").hexdigest()
            meta = uploads.init_upload(name="empty.wav", size=0, sha256=empty_hash)
            self.assertEqual(meta.received_bytes, 0)
            _, final_name = uploads.finish_sync(upload_id=meta.upload_id)
            target = uploads.files_dir / final_name
            self.assertTrue(target.is_file())
            self.assertEqual(target.stat().st_size, 0)

    def test_upload_same_path_overwrites_without_suffix_and_registry_cleanup(self):
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
            self.assertEqual(name1, name2)
            self.assertEqual((uploads.files_dir / name2).read_bytes(), data2)

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

    def test_folder_upload_preserves_relative_path(self):
        with tempfile.TemporaryDirectory() as tmp:
            uploads = UploadManager(Path(tmp))
            data = b"audio"
            meta = uploads.init_upload(
                name="dataset/speaker_a/take.wav",
                size=len(data),
                sha256=hashlib.sha256(data).hexdigest(),
            )
            uploads.write_chunk_sync(upload_id=meta.upload_id, offset=0, payload=data)
            _, final_name = uploads.finish_sync(upload_id=meta.upload_id)
            self.assertEqual(final_name, "dataset/speaker_a/take.wav")
            self.assertEqual(
                (uploads.files_dir / "dataset" / "speaker_a" / "take.wav").read_bytes(),
                data,
            )
            self.assertEqual(uploads.list_files()[0]["name"], final_name)
            uploads.delete_file(name=final_name)
            self.assertFalse((uploads.files_dir / "dataset").exists())

    def test_training_files_are_organized_by_model_and_speaker_without_overwrite(self):
        with tempfile.TemporaryDirectory() as tmp:
            uploads = UploadManager(Path(tmp))
            first = uploads.files_dir / "incoming" / "one.wav"
            second = uploads.files_dir / "other" / "one.wav"
            first.parent.mkdir(parents=True)
            second.parent.mkdir(parents=True)
            first.write_bytes(b"first")
            second.write_bytes(b"second")

            result = uploads.organize_training_files(
                model_name="my model",
                files=[
                    {"name": "incoming/one.wav", "speaker": "Alice"},
                    {"name": "other/one.wav", "speaker": "Alice"},
                ],
            )

            self.assertEqual(result["dataset_root"], "my model/dataset")
            self.assertEqual(
                [item["new_name"] for item in result["files"]],
                [
                    "my model/dataset/Alice/one.wav",
                    "my model/dataset/Alice/one_2.wav",
                ],
            )
            self.assertEqual((uploads.files_dir / result["files"][0]["new_name"]).read_bytes(), b"first")
            self.assertEqual((uploads.files_dir / result["files"][1]["new_name"]).read_bytes(), b"second")
            self.assertFalse((uploads.files_dir / "incoming").exists())
            self.assertFalse((uploads.files_dir / "other").exists())

    def test_rejects_traversal(self):
        with tempfile.TemporaryDirectory() as tmp:
            uploads = UploadManager(Path(tmp))
            with self.assertRaisesRegex(ValueError, "invalid filename"):
                uploads.init_upload(name="../safe.pth", size=1, sha256=hashlib.sha256(b"x").hexdigest())


if __name__ == "__main__":
    unittest.main()
