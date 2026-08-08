# RVC Streaming Server

Low-latency RVC inference server. The realtime transport uses protocol v2 with separate WebSocket endpoints:

- `/audio` — stream lifecycle, inference configuration, and little-endian float32 mono audio frames.
- `/control` — model/file management, uploads, logs, and health/control messages.

The split prevents large control responses or uploads from head-of-line blocking realtime audio.

## Run locally

The safe default only listens on localhost:

```bash
pixi run start
```

Default endpoint: `ws://127.0.0.1:8765`.

## Remote deployment

A non-loopback bind requires authentication. If a reverse proxy exposes a localhost-bound server remotely, configure authentication there as well (or set `RVC_STREAMING_TOKEN` on this server). Clear-text remote WebSocket is also rejected by default.

```bash
export RVC_STREAMING_BIND=0.0.0.0
export RVC_STREAMING_TOKEN='use-a-long-random-secret'
export RVC_TLS_CERT=/path/to/fullchain.pem
export RVC_TLS_KEY=/path/to/private-key.pem
pixi run start
```

Connect clients with `wss://host:8765`. The client must use the same `RVC_STREAMING_TOKEN` environment variable.

For a trusted private network only, TLS can be explicitly bypassed with `RVC_ALLOW_INSECURE_WS=1`. Do not use this on an untrusted network because the bearer token and audio would travel in clear text.

## Runtime limits

The server intentionally bounds realtime and upload queues so overload increases loss rather than unbounded latency or memory usage. Optional environment variables:

| Variable | Default | Purpose |
| --- | ---: | --- |
| `RVC_STREAMING_PORT` | `8765` | Listen port |
| `RVC_AUDIO_INPUT_QUEUE` | `8` | Maximum queued input audio frames; oldest frames are dropped first |
| `RVC_AUDIO_OUTPUT_QUEUE` | `1` | Maximum queued inference blocks; oldest output is dropped first |
| `RVC_MAX_INPUT_BACKLOG_MS` | `200` | Maximum queued input media duration; oversized/old input is trimmed oldest-first |
| `RVC_MODEL_CACHE_SIZE` | `4` | Shared process-wide voice model cache entries |
| `RVC_BASE_MODEL_CACHE_SIZE` | `2` | Process-wide HuBERT/RMVPE LRU cache entries |
| `RVC_INDEX_CACHE_SIZE` | `4` | Process-wide FAISS index LRU cache entries |
| `RVC_MAX_UPLOAD_BYTES` | `2147483648` | Maximum size of one uploaded file |
| `RVC_MAX_ACTIVE_UPLOADS` | `8` | Maximum resumable partial uploads |
| `RVC_MAX_RESERVED_UPLOAD_BYTES` | `4294967296` | Aggregate reservation cap for partial uploads |
| `RVC_STALE_UPLOAD_SECONDS` | `86400` | Cleanup age for incomplete uploads on startup |

## Model security

RVC and RMVPE checkpoints are loaded with PyTorch `weights_only=True`. Uploaded model files must therefore be tensor/config checkpoints rather than arbitrary pickle programs.

Some historical fairseq HuBERT checkpoints embed pickled configuration objects. Unsafe legacy loading is disabled by default. If a specific HuBERT checkpoint is trusted and cannot be converted, set:

```bash
export RVC_ALLOW_LEGACY_HUBERT_PICKLE=1
```

Only enable that compatibility mode for a checkpoint whose origin you trust.

## Realtime behavior

Protocol v2 carries a stream `session_id`, sequence number, sample rate, media timestamp, and discontinuity flag. Starting a new stream resets pending input, pitch history, noise-reduction overlap, timestamps, and SOLA state without reloading the model.

Input and output queues are latest-wins. If inference or the network falls behind, stale audio is discarded and the next output is marked discontinuous so the client can fast-forward instead of preserving seconds of old speech.

Configuration updates are classified internally. Hot controls such as pitch, formant, RMS mix, silence gate, and index-rate adjustments do not rebuild the stream buffers or rerun model warmup. Buffer-layout/model changes reset only the state they actually require.

The server sends one WebSocket audio frame per inference block. The old `stream_chunk_ms` post-inference slicing behavior was removed because cutting an already-computed block into small burst packets did not reduce algorithmic latency.

## Validation

Static tests that do not require CUDA can be run with:

```bash
PYTHONPATH=. python -m unittest discover -s tests -v
python -m compileall -q .
```

Actual inference performance should be benchmarked on the deployment GPU because HuBERT/F0/synthesizer timings and the optimal block/context lengths depend strongly on the GPU and model.
