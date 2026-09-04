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

## Private local process mode

The desktop client can start an installed server directly from its connection panel. Select **本地** and open **设置**. **从 GitHub 下载** retrieves the `master` branch source archive without Git history, while **下载依赖** runs `pixi install --environment default --locked` and streams its output in the settings window. Choose an existing installation or complete those two steps, run **检查依赖**, save, and use **启动并连接**. The default directory is `localServer` beside the client executable. The directory must contain `server.py`, `pixi.toml`, and an installed `.pixi/envs/default` environment. The separate dependency check uses `--no-install --frozen`; it does not install packages or update `pixi.lock`.

In this mode the client invokes:

```bash
python server.py --stdio
```

Control and audio messages are multiplexed over the child process's redirected standard-input and standard-output pipes. No TCP/UDP port is opened, and the anonymous pipe handles are inherited only by the client and its server child process. Closing or disconnecting the client closes the pipes and stops the child server. `pixi run start-local` exposes the same server entry point for transport diagnostics; it expects a framed parent process rather than an interactive terminal.

In client-managed local mode, `RVC_DATA_DIR` points to `data/server` beside the client executable. Uploaded files, registries, training state, Server logs, bytecode, and external model caches therefore stay under the executable's `data` tree instead of AppData or the Server source directory.

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
| `RVC_CUDA_GRAPH` | `auto` | CUDA Graph acceleration (`auto`, `1`, or `0`); capture failures fall back to eager inference |
| `RVC_CUDA_GRAPH_CACHE_SIZE` | `12` | Maximum captured static-shape calls per model component |
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

Inference still runs in full blocks, but each completed block is emitted as clock-paced `stream_chunk_ms` packets (20 ms by default). Pacing is deliberate: unlike burst slicing, it gives the client a fine-grained arrival stream for adaptive playout while preserving the inference algorithm and its timestamps. If the sender falls behind, it restarts from the current clock instead of sending a catch-up burst.

## Validation

Static tests that do not require CUDA can be run with:

```bash
PYTHONPATH=. python -m unittest discover -s tests -v
python -m compileall -q .
```

Actual inference performance should be benchmarked on the deployment GPU because HuBERT/F0/synthesizer timings and the optimal block/context lengths depend strongly on the GPU and model.

## Native training

The client training page runs a server-managed RVC v2 workflow. Upload individual audio files, a batch of files, or a folder through the existing file manager, configure the active HuBERT and RMVPE slots, then select the server-side audio files and speaker labels on the training page. The worker validates every selected file, creates features and F0, trains an inference-compatible generator, builds a FAISS index, and registers the completed voice model automatically.

Training and realtime inference compete for the same GPU. Stop live conversion before starting a training job. Optional compatible generator and discriminator checkpoints can be uploaded and supplied by filename; without them, training starts from random initialization and generally requires more data and epochs.

### Optional PyMSS preprocessing

The training page can optionally remove accompaniment, reduce reverb, or extract the lead melody before slicing and feature extraction. Upload a compatible PyMSS model checkpoint and its YAML configuration, then bind them to the `PyMSS` slots in the model/file page.

PyMSS 2.0 currently uses the Torch 2.7 generation, so it is kept outside the Torch 2.11 realtime environment. Install it once with:

```bash
pixi install -e pymss
```

The server discovers `.pixi/envs/pymss` automatically. A separately managed environment can instead be selected with `RVC_PYMSS_PYTHON=/path/to/python`, provided that interpreter contains `pymss==2.0.14`, `pymss-core==0.1.4`, and a compatible Torch build. The selected weight and YAML must belong to the same model. The original RVC WebUI's vocal and dereverb presets use BS-Roformer for vocal extraction and Mel-Band Roformer for dereverb/lead-melody extraction.
