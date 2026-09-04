import asyncio
import concurrent.futures
import io
import json
import os
import subprocess
import struct
import sys
import tempfile
import unittest
from pathlib import Path

from server import (
    _STDIO_CHANNEL_AUDIO,
    _STDIO_CHANNEL_CONTROL,
    _STDIO_HEADER,
    _STDIO_KIND_BINARY,
    _STDIO_KIND_TEXT,
    _STDIO_MAGIC,
    _StdioEndpoint,
    _StdioFrameWriter,
    _read_stdio_frame,
)


class _ChunkedReader(io.BytesIO):
    def read(self, size=-1):
        if size > 0:
            size = min(size, 3)
        return super().read(size)


class StdioFrameTests(unittest.TestCase):
    def test_reads_partial_pipe_chunks(self):
        payload = b'{"command":"ping"}'
        encoded = _STDIO_HEADER.pack(
            _STDIO_MAGIC,
            _STDIO_CHANNEL_CONTROL,
            _STDIO_KIND_TEXT,
            len(payload),
        ) + payload

        self.assertEqual(
            (_STDIO_CHANNEL_CONTROL, _STDIO_KIND_TEXT, payload),
            _read_stdio_frame(_ChunkedReader(encoded)),
        )

    def test_rejects_oversized_frame_before_reading_payload(self):
        encoded = struct.pack("<4sBBI", _STDIO_MAGIC, 0, 1, 2 * 1024 * 1024 + 1)
        with self.assertRaisesRegex(ValueError, "too large"):
            _read_stdio_frame(io.BytesIO(encoded))


class StdioEndpointTests(unittest.IsolatedAsyncioTestCase):
    async def test_preserves_text_and_binary_message_types(self):
        endpoint = _StdioEndpoint(
            _STDIO_CHANNEL_AUDIO,
            "/audio",
            _StdioFrameWriter(io.BytesIO()),
        )
        endpoint.feed(_STDIO_KIND_TEXT, "配置".encode("utf-8"))
        endpoint.feed(_STDIO_KIND_BINARY, b"\x01\x02")

        self.assertEqual("配置", await endpoint.__anext__())
        self.assertEqual(b"\x01\x02", await endpoint.__anext__())


class StdioProcessIntegrationTests(unittest.TestCase):
    def test_stdio_server_handles_control_message_and_exits_on_eof(self):
        root = Path(__file__).resolve().parents[1]
        with tempfile.TemporaryDirectory() as data_directory:
            environment = os.environ.copy()
            environment["RVC_DATA_DIR"] = data_directory
            process = subprocess.Popen(
                [sys.executable, str(root / "server.py"), "--stdio"],
                cwd=root,
                env=environment,
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
            )
            self.assertIsNotNone(process.stdin)
            self.assertIsNotNone(process.stdout)
            self.assertIsNotNone(process.stderr)

            try:
                with concurrent.futures.ThreadPoolExecutor(max_workers=1) as executor:
                    ready = executor.submit(_read_stdio_frame, process.stdout).result(timeout=60)
                    self.assertEqual((255, 4, b""), ready)

                    request = json.dumps({"command": "ping", "ts": 123}).encode("utf-8")
                    process.stdin.write(
                        _STDIO_HEADER.pack(
                            _STDIO_MAGIC,
                            _STDIO_CHANNEL_CONTROL,
                            _STDIO_KIND_TEXT,
                            len(request),
                        )
                        + request
                    )
                    process.stdin.flush()
                    response = executor.submit(_read_stdio_frame, process.stdout).result(timeout=15)
                    self.assertEqual(_STDIO_CHANNEL_CONTROL, response[0])
                    self.assertEqual(_STDIO_KIND_TEXT, response[1])
                    body = json.loads(response[2])
                    self.assertEqual("pong", body["type"])
                    self.assertEqual(123, body["client_ts"])

                process.stdin.close()
                process.wait(timeout=15)
                self.assertEqual(0, process.returncode, process.stderr.read().decode("utf-8", errors="replace"))
            finally:
                if process.poll() is None:
                    process.kill()
                    process.wait(timeout=5)
                if process.stdin and not process.stdin.closed:
                    process.stdin.close()
                if process.stdout:
                    process.stdout.close()
                if process.stderr:
                    process.stderr.close()


if __name__ == "__main__":
    unittest.main()
