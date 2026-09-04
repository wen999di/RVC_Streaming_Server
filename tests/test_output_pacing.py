import asyncio
import contextlib
import time
import unittest

from audio_protocol import OUTPUT_HEADER
from server import RealtimeAudioSession, _clamp_stream_chunk_ms, _iter_output_slices


class _RecordingWebSocket:
    def __init__(self, expected_count):
        self.frames = []
        self.complete = asyncio.Event()
        self.expected_count = expected_count

    async def send(self, frame):
        self.frames.append(frame)
        if len(self.frames) >= self.expected_count:
            self.complete.set()


class _FakeCore:
    passthrough = True
    model_path = ""


class _FakeProcessor:
    core = _FakeCore()

    def update_config(self, _cfg):
        return {"buffer_layout": False, "model_runtime": False}

    def reset_stream_state(self):
        return None


class OutputPacingTests(unittest.TestCase):
    def test_250ms_block_is_split_into_paced_20ms_packets(self):
        sample_rate = 16000
        sample_count = sample_rate // 4
        payload = bytes(sample_count * 4)
        base_timestamp_ns = 1_000_000_000

        slices = list(_iter_output_slices(payload, base_timestamp_ns, 20, sample_rate))

        self.assertEqual(len(slices), 13)
        self.assertEqual([item[2] for item in slices[:-1]], [320] * 12)
        self.assertEqual(slices[-1][2], 160)
        self.assertEqual(b"".join(item[0] for item in slices), payload)
        self.assertEqual(slices[0][1], base_timestamp_ns)
        self.assertEqual(slices[-1][1], base_timestamp_ns + 3840 * 62_500)

    def test_unknown_timestamp_remains_unknown(self):
        slices = list(_iter_output_slices(bytes(640 * 4), 0, 20))
        self.assertTrue(slices)
        self.assertTrue(all(timestamp_ns == 0 for _, timestamp_ns, _ in slices))

    def test_chunk_setting_is_bounded(self):
        self.assertEqual(_clamp_stream_chunk_ms(1), 10)
        self.assertEqual(_clamp_stream_chunk_ms(20), 20)
        self.assertEqual(_clamp_stream_chunk_ms(500), 120)
        self.assertEqual(_clamp_stream_chunk_ms("invalid"), 20)


class OutputQueueTelemetryTests(unittest.IsolatedAsyncioTestCase):
    async def test_audio_connection_requires_a_successful_configuration(self):
        session = RealtimeAudioSession(_RecordingWebSocket(expected_count=1), _FakeProcessor())
        try:
            self.assertFalse(session.configuration_received)
            await session.apply_config({"stream_chunk_ms": 20})
            self.assertTrue(session.configuration_received)
        finally:
            session._worker_task.cancel()
            session._sender_task.cancel()
            for task in (session._worker_task, session._sender_task):
                with contextlib.suppress(asyncio.CancelledError):
                    await task

    async def test_paced_slices_reuse_block_queue_wait(self):
        websocket = _RecordingWebSocket(expected_count=3)
        session = object.__new__(RealtimeAudioSession)
        session.websocket = websocket
        session.active_session_id = 1
        session.output_sequence = 0
        session.output_slice_ms = 20
        session._next_output_send_time = 0.0
        session._output_epoch = 0
        session.output_queue = asyncio.Queue()

        task = asyncio.create_task(session._sender_loop())
        try:
            payload = bytes(16_000 * 60 // 1000 * 4)
            session.output_queue.put_nowait(
                (1, 0, 1_000_000_000, 40, 3, 0, payload, time.perf_counter())
            )
            await asyncio.wait_for(websocket.complete.wait(), timeout=1.0)

            queue_values = [OUTPUT_HEADER.unpack_from(frame)[10] for frame in websocket.frames]
            self.assertEqual(len(queue_values), 3)
            self.assertTrue(all(value == queue_values[0] for value in queue_values))
        finally:
            task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await task


if __name__ == "__main__":
    unittest.main()
