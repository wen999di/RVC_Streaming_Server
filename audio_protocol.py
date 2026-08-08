from __future__ import annotations

from dataclasses import dataclass
import struct

PROTOCOL_VERSION = 2
INPUT_MAGIC = b"RVCA"
OUTPUT_MAGIC = b"RVCO"
INPUT_HEADER = struct.Struct(">4sBBHQIIQ")
OUTPUT_HEADER = struct.Struct(">4sBBHQIIQHHHH")

FLAG_DISCONTINUITY = 0x01
MAX_AUDIO_PAYLOAD_BYTES = 16000 * 4 // 2  # 500 ms


@dataclass(frozen=True)
class AudioInputFrame:
    session_id: int
    sequence: int
    sample_rate: int
    timestamp_ns: int
    flags: int
    payload: bytes


def parse_audio_input_frame(frame: bytes) -> AudioInputFrame | None:
    if len(frame) < INPUT_HEADER.size:
        return None
    magic, version, flags, header_len, session_id, sequence, sample_rate, timestamp_ns = INPUT_HEADER.unpack_from(frame)
    if magic != INPUT_MAGIC:
        return None
    if version != PROTOCOL_VERSION:
        raise ValueError(f"unsupported_audio_protocol:{version}")
    if header_len != INPUT_HEADER.size:
        raise ValueError("invalid_audio_header_length")
    if sample_rate != 16000:
        raise ValueError(f"unsupported_sample_rate:{sample_rate}")
    payload = frame[header_len:]
    if not payload or len(payload) % 4 != 0:
        raise ValueError("invalid_audio_payload")
    if len(payload) > MAX_AUDIO_PAYLOAD_BYTES:
        raise ValueError("audio_payload_too_large")
    return AudioInputFrame(
        session_id=int(session_id),
        sequence=int(sequence),
        sample_rate=int(sample_rate),
        timestamp_ns=int(timestamp_ns),
        flags=int(flags),
        payload=payload,
    )


def build_audio_output_frame(
    *,
    session_id: int,
    sequence: int,
    sample_rate: int,
    timestamp_ns: int,
    proc_ms: int,
    input_queue_ms: int,
    output_queue_ms: int,
    flags: int,
    payload: bytes,
) -> bytes:
    def u16(value: int) -> int:
        return max(0, min(65535, int(value)))

    header = OUTPUT_HEADER.pack(
        OUTPUT_MAGIC,
        PROTOCOL_VERSION,
        int(flags) & 0xFF,
        OUTPUT_HEADER.size,
        int(session_id) & 0xFFFFFFFFFFFFFFFF,
        int(sequence) & 0xFFFFFFFF,
        int(sample_rate),
        max(0, int(timestamp_ns)),
        u16(proc_ms),
        u16(input_queue_ms),
        u16(output_queue_ms),
        0,
    )
    return header + payload
