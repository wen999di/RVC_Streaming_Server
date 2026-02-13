import asyncio
import hashlib
import json
import os
import struct
import uuid

import websockets


FILE_MAGIC = b"RVCFILE1"


def sha256_bytes(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()


def build_chunk(upload_id: str, offset: int, payload: bytes) -> bytes:
    u = uuid.UUID(upload_id)
    header = bytearray()
    header += FILE_MAGIC
    header += bytes([1])
    header += u.bytes
    header += struct.pack(">Q", offset)
    header += struct.pack(">I", len(payload))
    return bytes(header) + payload


async def recv_json(ws):
    while True:
        msg = await ws.recv()
        if isinstance(msg, str):
            return json.loads(msg)


async def main():
    uri = "ws://127.0.0.1:8765"
    name = "test_model.pt"
    content = os.urandom(2 * 1024 * 1024 + 12345)
    size = len(content)
    sha = sha256_bytes(content)

    voice_pth_name = "voice_test.pth"
    voice_pth_content = os.urandom(512 * 1024 + 7)
    voice_pth_sha = sha256_bytes(voice_pth_content)
    voice_index_name = "voice_test.index"
    voice_index_content = os.urandom(128 * 1024 + 3)
    voice_index_sha = sha256_bytes(voice_index_content)

    async with websockets.connect(uri, max_size=None) as ws:
        await ws.send(json.dumps({"command": "upload_init", "name": name, "size": size, "sha256": sha}))
        ready = await recv_json(ws)
        assert ready["type"] == "upload_ready", ready
        upload_id = ready["upload_id"]
        received = int(ready["received_bytes"])
        assert received == 0

        halfway = size // 2
        offset = 0
        chunk = 256 * 1024
        while offset < halfway:
            part = content[offset : min(offset + chunk, halfway)]
            await ws.send(build_chunk(upload_id, offset, part))
            _ = await recv_json(ws)
            offset += len(part)

    async with websockets.connect(uri, max_size=None) as ws:
        await ws.send(json.dumps({"command": "upload_init", "name": name, "size": size, "sha256": sha}))
        ready2 = await recv_json(ws)
        assert ready2["type"] == "upload_ready", ready2
        upload_id2 = ready2["upload_id"]
        assert upload_id2 == upload_id
        offset = int(ready2["received_bytes"])

        while offset < size:
            part = content[offset : min(offset + chunk, size)]
            await ws.send(build_chunk(upload_id, offset, part))
            _ = await recv_json(ws)
            offset += len(part)

        await ws.send(json.dumps({"command": "upload_finish", "upload_id": upload_id}))
        done = await recv_json(ws)
        assert done["type"] == "upload_done", done
        final_name = done["name"]

        await ws.send(json.dumps({"command": "upload_init", "name": voice_pth_name, "size": len(voice_pth_content), "sha256": voice_pth_sha}))
        ready_pth = await recv_json(ws)
        assert ready_pth["type"] == "upload_ready", ready_pth
        pth_upload_id = ready_pth["upload_id"]
        offset = int(ready_pth["received_bytes"])
        chunk = 256 * 1024
        while offset < len(voice_pth_content):
            part = voice_pth_content[offset : min(offset + chunk, len(voice_pth_content))]
            await ws.send(build_chunk(pth_upload_id, offset, part))
            _ = await recv_json(ws)
            offset += len(part)
        await ws.send(json.dumps({"command": "upload_finish", "upload_id": pth_upload_id}))
        done_pth = await recv_json(ws)
        assert done_pth["type"] == "upload_done", done_pth
        final_pth_name = done_pth["name"]

        await ws.send(json.dumps({"command": "upload_init", "name": voice_index_name, "size": len(voice_index_content), "sha256": voice_index_sha}))
        ready_index = await recv_json(ws)
        assert ready_index["type"] == "upload_ready", ready_index
        index_upload_id = ready_index["upload_id"]
        offset = int(ready_index["received_bytes"])
        while offset < len(voice_index_content):
            part = voice_index_content[offset : min(offset + chunk, len(voice_index_content))]
            await ws.send(build_chunk(index_upload_id, offset, part))
            _ = await recv_json(ws)
            offset += len(part)
        await ws.send(json.dumps({"command": "upload_finish", "upload_id": index_upload_id}))
        done_index = await recv_json(ws)
        assert done_index["type"] == "upload_done", done_index
        final_index_name = done_index["name"]

        await ws.send(json.dumps({"command": "files_list"}))
        files = await recv_json(ws)
        assert files["type"] == "files_list", files
        assert any(f["name"] == final_name for f in files["files"])
        assert any(f["name"] == final_pth_name for f in files["files"])
        assert any(f["name"] == final_index_name for f in files["files"])

        await ws.send(json.dumps({"command": "model_set_slot", "slot": "hubert_base", "filename": final_name}))
        bound = await recv_json(ws)
        assert bound["type"] == "model_slot_updated", bound

        await ws.send(json.dumps({"command": "model_list_slots"}))
        slots = await recv_json(ws)
        assert slots["type"] == "model_slots", slots
        assert slots["slots"]["hubert_base"]["active"] == final_name

        await ws.send(json.dumps({"command": "voice_model_add", "name": "voice_test", "pth": final_pth_name, "index": final_index_name}))
        voice = await recv_json(ws)
        assert voice["type"] == "voice_models", voice
        models = voice["voice"]["models"]
        assert any(m["name"] == "voice_test" and m["pth"] == final_pth_name for m in models)

        renamed_pth = "voice_test_renamed.pth"
        await ws.send(json.dumps({"command": "files_rename", "old_name": final_pth_name, "new_name": renamed_pth}))
        renamed_evt = await recv_json(ws)
        assert renamed_evt["type"] == "files_renamed", renamed_evt
        assert renamed_evt["new_name"] == renamed_pth
        _ = await recv_json(ws)  # model_slots
        voice_after = await recv_json(ws)
        assert voice_after["type"] == "voice_models", voice_after
        models_after = voice_after["voice"]["models"]
        assert any(m["name"] == "voice_test" and m["pth"] == renamed_pth for m in models_after)

        await ws.send(json.dumps({"command": "voice_model_list"}))
        voice2 = await recv_json(ws)
        assert voice2["type"] == "voice_models", voice2

    print("OK")


if __name__ == "__main__":
    asyncio.run(main())
