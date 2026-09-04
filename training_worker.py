from __future__ import annotations

import json
import math
import os
import re
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
import torchaudio
from torch.utils.data import DataLoader, Dataset

from hubert import load_hubert
from rmvpe import RMVPE
from training_models import (
    MultiPeriodDiscriminator,
    TrainableSynthesizerV2,
    discriminator_loss,
    feature_matching_loss,
    generator_loss,
    kl_loss,
    mel_spectrogram,
    mel_spectrogram_from_linear,
    slice_segments,
    spectrogram,
)


def emit(**event) -> None:
    print(json.dumps(event, ensure_ascii=False), flush=True)


def safe_name(value: str) -> str:
    cleaned = re.sub(r"[^0-9A-Za-z\u4e00-\u9fff._-]+", "_", value.strip())
    return cleaned.strip("._")[:80] or "trained_voice"


def stage(name: str, progress: float, message: str, **extra) -> None:
    emit(state="running", stage=name, progress=max(0.0, min(1.0, progress)), message=message, **extra)


def model_shape(sample_rate: int) -> dict:
    if sample_rate == 32000:
        return {
            "n_fft": 1024, "hop": 320, "window": 1024, "n_mels": 80,
            "segment_samples": 12800, "upsample_rates": [10, 8, 2, 2],
            "upsample_kernels": [20, 16, 4, 4],
        }
    if sample_rate == 40000:
        return {
            "n_fft": 2048, "hop": 400, "window": 2048, "n_mels": 125,
            "segment_samples": 12800, "upsample_rates": [10, 10, 2, 2],
            "upsample_kernels": [16, 16, 4, 4],
        }
    if sample_rate == 48000:
        return {
            "n_fft": 2048, "hop": 480, "window": 2048, "n_mels": 128,
            "segment_samples": 17280, "upsample_rates": [12, 10, 2, 2],
            "upsample_kernels": [24, 20, 4, 4],
        }
    raise ValueError(f"unsupported sample rate: {sample_rate}")


def preprocess_audio_files(config: dict) -> list[dict]:
    audio_files = config.get("audio_files")
    if not isinstance(audio_files, list) or not audio_files:
        raise ValueError("没有配置训练音频")
    mode = str(config.get("preprocess") or "none")
    if mode == "none":
        return audio_files

    work_dir = Path(config["work_dir"])
    request = {
        "audio_files": audio_files,
        "output_dir": str(work_dir / "pymss_audio"),
        "model_type": config["pymss_model_type"],
        "model_path": config["pymss_weight_path"],
        "config_path": config["pymss_config_path"],
        "stem": config["pymss_stem"],
        "device": "auto",
    }
    request_path = work_dir / "pymss_job.json"
    with open(request_path, "w", encoding="utf-8") as handle:
        json.dump(request, handle, ensure_ascii=False, indent=2)

    command = [
        str(config["pymss_python"]),
        "-u",
        str(Path(__file__).with_name("pymss_preprocess_worker.py")),
        str(request_path),
    ]
    child_env = os.environ.copy()
    pymss_environment = Path(str(config["pymss_python"])).resolve().parent
    runtime_paths = [
        pymss_environment,
        pymss_environment / "Scripts",
        pymss_environment / "Library" / "bin",
        pymss_environment / "bin",
    ]
    child_env["PATH"] = os.pathsep.join(
        [str(path) for path in runtime_paths if path.is_dir()]
        + [child_env.get("PATH", "")]
    )
    child_env["PYTHONUTF8"] = "1"
    creationflags = getattr(subprocess, "CREATE_NO_WINDOW", 0)
    process = subprocess.Popen(
        command,
        cwd=str(Path(__file__).parent),
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        encoding="utf-8",
        errors="replace",
        bufsize=1,
        creationflags=creationflags,
        env=child_env,
    )
    result: list[dict] | None = None
    last_error = ""
    assert process.stdout is not None
    for line in process.stdout:
        text = line.strip()
        if not text:
            continue
        try:
            event = json.loads(text)
        except json.JSONDecodeError:
            continue
        if not isinstance(event, dict):
            continue
        event_type = event.get("type")
        if event_type == "pymss_status":
            progress = float(event.get("progress") or 0.0)
            stage("preprocess", 0.02 + 0.16 * progress, str(event.get("message") or "正在处理训练音频"))
        elif event_type == "pymss_result" and isinstance(event.get("audio_files"), list):
            result = event["audio_files"]
        elif event_type == "pymss_error":
            last_error = str(event.get("error") or "PyMSS 处理失败")
    code = process.wait()
    if code != 0 or not result:
        raise RuntimeError(last_error or f"PyMSS 前处理进程退出码 {code}")
    return result


def decode_segments(path: Path, target_rate: int):
    waveform, source_rate = torchaudio.load(str(path))
    if waveform.numel() == 0:
        return
    waveform = waveform.float().mean(dim=0)
    if int(source_rate) != target_rate:
        waveform = torchaudio.functional.resample(waveform, int(source_rate), target_rate)
    waveform = torch.nan_to_num(waveform)
    peak = float(waveform.abs().max().item()) if waveform.numel() else 0.0
    if peak < 1e-5:
        return
    waveform = waveform * min(1.0, 0.95 / peak)
    maximum = target_rate * 12
    minimum = target_rate * 2
    overlap = target_rate // 4
    start = 0
    while start < waveform.shape[0]:
        end = min(waveform.shape[0], start + maximum)
        piece = waveform[start:end]
        if piece.shape[0] >= minimum:
            rms = float(torch.sqrt(torch.mean(piece * piece) + 1e-12).item())
            if rms >= 1e-4:
                yield piece.contiguous()
        if end >= waveform.shape[0]:
            break
        start = max(start + 1, end - overlap)


def f0_to_coarse(f0: torch.Tensor) -> torch.Tensor:
    f0 = f0.float().clamp_min(0)
    mel = 1127.0 * torch.log1p(f0 / 700.0)
    minimum = 1127.0 * math.log1p(50.0 / 700.0)
    maximum = 1127.0 * math.log1p(1100.0 / 700.0)
    active = mel > 0
    mel[active] = (mel[active] - minimum) * 254.0 / (maximum - minimum) + 1.0
    return torch.round(mel.clamp(1.0, 255.0)).long()


def prepare_examples(config: dict, shape: dict, device: torch.device) -> tuple[list[Path], list[dict]]:
    work_dir = Path(config["work_dir"])
    examples_dir = work_dir / "examples"
    examples_dir.mkdir(parents=True, exist_ok=True)
    audio_files = preprocess_audio_files(config)
    speakers = sorted({str(item["speaker"]) for item in audio_files})
    speaker_ids = {name: index for index, name in enumerate(speakers)}
    speaker_info = [{"id": index, "name": name} for name, index in speaker_ids.items()]
    if len(speaker_info) > 109:
        raise ValueError("说话人数量不能超过 109")

    is_half = device.type == "cuda"
    feature_start = 0.20 if str(config.get("preprocess") or "none") != "none" else 0.08
    stage("features", feature_start, "正在加载 HuBERT 与 RMVPE", speaker_count=len(speakers))
    hubert = load_hubert(config["hubert_path"], device, is_half)
    rmvpe = RMVPE(config["rmvpe_path"], is_half=is_half, device=device)
    prepared: list[Path] = []
    index_features: dict[int, list[np.ndarray]] = {item["id"]: [] for item in speaker_info}
    total_inputs = len(audio_files)
    example_index = 0

    with torch.inference_mode():
        for file_index, source_item in enumerate(audio_files):
            source = Path(source_item["path"])
            speaker = speaker_ids[str(source_item["speaker"])]
            for audio in decode_segments(source, int(config["sample_rate"])):
                audio_16k = torchaudio.functional.resample(
                    audio.unsqueeze(0), int(config["sample_rate"]), 16000
                ).squeeze(0)
                source_tensor = audio_16k.to(
                    device=device,
                    dtype=torch.float16 if is_half else torch.float32,
                ).unsqueeze(0)
                features = hubert.extract_features(
                    source=source_tensor,
                    padding_mask=None,
                    output_layer=12,
                )[0].float()
                features = torch.cat((features, features[:, -1:, :]), dim=1)
                features = features.repeat_interleave(2, dim=1).squeeze(0)
                continuous = rmvpe.infer_from_audio(audio_16k.to(device)).float().view(-1)
                linear = spectrogram(
                    audio.to(device).unsqueeze(0), shape["n_fft"], shape["hop"], shape["window"]
                ).squeeze(0).float()
                frames = min(features.shape[0], continuous.shape[0], linear.shape[1])
                if frames < max(20, shape["segment_samples"] // shape["hop"]):
                    continue
                features = features[:frames].cpu()
                continuous = continuous[:frames].cpu()
                linear = linear[:, :frames].cpu()
                audio = audio[: frames * shape["hop"]].cpu()
                item = {
                    "audio": audio,
                    "features": features,
                    "pitch": f0_to_coarse(continuous),
                    "pitchf": continuous,
                    "spectrogram": linear,
                    "speaker": speaker,
                }
                output = examples_dir / f"{example_index:06d}.pt"
                torch.save(item, output)
                prepared.append(output)
                index_features[speaker].append(features.numpy().astype(np.float32, copy=False))
                example_index += 1
            stage(
                "features",
                feature_start + (0.35 - feature_start) * ((file_index + 1) / total_inputs),
                f"已处理 {file_index + 1}/{total_inputs} 个音频文件",
                speaker_count=len(speakers),
            )

    del hubert, rmvpe
    if device.type == "cuda":
        torch.cuda.empty_cache()
    if not prepared:
        raise ValueError("没有生成可训练片段；请检查音频长度和音量")
    for speaker_id, values in index_features.items():
        if values:
            np.save(
                work_dir / f"index_features_spkid{speaker_id}.npy",
                np.concatenate(values, axis=0),
            )
    with open(work_dir / "speaker_info.json", "w", encoding="utf-8") as handle:
        json.dump(speaker_info, handle, ensure_ascii=False, indent=2)
    return prepared, speaker_info


class PreparedDataset(Dataset):
    def __init__(self, paths: list[Path]):
        self.paths = paths

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, index):
        return torch.load(self.paths[index], map_location="cpu", weights_only=True)


def collate(batch: list[dict]):
    batch.sort(key=lambda item: item["features"].shape[0], reverse=True)
    count = len(batch)
    max_frames = max(item["features"].shape[0] for item in batch)
    max_audio = max(item["audio"].shape[0] for item in batch)
    freq = batch[0]["spectrogram"].shape[0]
    phone = torch.zeros(count, max_frames, 768)
    pitch = torch.zeros(count, max_frames, dtype=torch.long)
    pitchf = torch.zeros(count, max_frames)
    spec = torch.zeros(count, freq, max_frames)
    audio = torch.zeros(count, 1, max_audio)
    lengths = torch.zeros(count, dtype=torch.long)
    speakers = torch.zeros(count, dtype=torch.long)
    for row, item in enumerate(batch):
        frames = item["features"].shape[0]
        samples = item["audio"].shape[0]
        phone[row, :frames] = item["features"]
        pitch[row, :frames] = item["pitch"]
        pitchf[row, :frames] = item["pitchf"]
        spec[row, :, :frames] = item["spectrogram"]
        audio[row, 0, :samples] = item["audio"]
        lengths[row] = frames
        speakers[row] = int(item["speaker"])
    return phone, lengths, pitch, pitchf, spec, audio, speakers


def generator_args(shape: dict, sample_rate: int, speakers: int) -> list:
    return [
        shape["n_fft"] // 2 + 1,
        shape["segment_samples"] // shape["hop"],
        192,
        192,
        768,
        2,
        6,
        3,
        0.0,
        "1",
        [3, 7, 11],
        [[1, 3, 5], [1, 3, 5], [1, 3, 5]],
        shape["upsample_rates"],
        512,
        shape["upsample_kernels"],
        speakers,
        256,
        sample_rate,
    ]


def load_compatible(module: torch.nn.Module, path: str) -> None:
    if not path:
        return
    checkpoint = torch.load(path, map_location="cpu", weights_only=True)
    state = checkpoint.get("model", checkpoint.get("weight", checkpoint))
    if not isinstance(state, dict):
        raise ValueError(f"预训练权重格式无效: {Path(path).name}")
    current = module.state_dict()
    usable = {
        key: value for key, value in state.items()
        if key in current and torch.is_tensor(value) and value.shape == current[key].shape
    }
    module.load_state_dict(usable, strict=False)


def train(config: dict, shape: dict, examples: list[Path], speaker_info: list[dict], device: torch.device):
    sample_rate = int(config["sample_rate"])
    args = generator_args(shape, sample_rate, len(speaker_info))
    generator = TrainableSynthesizerV2(*args, is_half=device.type == "cuda").to(device)
    discriminator = MultiPeriodDiscriminator().to(device)
    load_compatible(generator, config.get("pretrained_g", ""))
    load_compatible(discriminator, config.get("pretrained_d", ""))

    optimizer_g = torch.optim.AdamW(
        generator.parameters(), lr=float(config["learning_rate"]), betas=(0.8, 0.99), eps=1e-9
    )
    optimizer_d = torch.optim.AdamW(
        discriminator.parameters(), lr=float(config["learning_rate"]), betas=(0.8, 0.99), eps=1e-9
    )
    scheduler_g = torch.optim.lr_scheduler.ExponentialLR(optimizer_g, gamma=0.999875)
    scheduler_d = torch.optim.lr_scheduler.ExponentialLR(optimizer_d, gamma=0.999875)
    amp_enabled = device.type == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=amp_enabled)
    loader = DataLoader(
        PreparedDataset(examples),
        batch_size=int(config["batch_size"]),
        shuffle=True,
        collate_fn=collate,
        num_workers=0,
        pin_memory=amp_enabled,
    )
    epochs = int(config["epochs"])
    segment_frames = shape["segment_samples"] // shape["hop"]
    global_step = 0
    checkpoint_dir = Path(config["work_dir"]) / "checkpoints"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    for epoch in range(1, epochs + 1):
        generator.train()
        discriminator.train()
        running = 0.0
        for batch_index, packed in enumerate(loader, 1):
            phone, lengths, pitch, pitchf, linear, audio, speakers = (
                value.to(device, non_blocking=True) for value in packed
            )
            with torch.amp.autocast("cuda", enabled=amp_enabled):
                generated, starts, _, spec_mask, latent_values = generator(
                    phone, lengths, pitch, pitchf, linear, lengths, speakers
                )
                real_audio = slice_segments(
                    audio, starts * shape["hop"], shape["segment_samples"]
                )
                real_scores, fake_scores, _, _ = discriminator(real_audio, generated.detach())
                loss_d = discriminator_loss(real_scores, fake_scores)
            optimizer_d.zero_grad(set_to_none=True)
            scaler.scale(loss_d).backward()
            scaler.unscale_(optimizer_d)
            torch.nn.utils.clip_grad_norm_(discriminator.parameters(), 100.0)
            scaler.step(optimizer_d)

            with torch.amp.autocast("cuda", enabled=amp_enabled):
                real_scores, fake_scores, real_maps, fake_maps = discriminator(real_audio, generated)
                original_mel = mel_spectrogram_from_linear(
                    linear,
                    sample_rate=sample_rate,
                    n_fft=shape["n_fft"],
                    n_mels=shape["n_mels"],
                )
                original_mel = slice_segments(original_mel, starts, segment_frames)
            with torch.amp.autocast("cuda", enabled=False):
                generated_mel = mel_spectrogram(
                    generated.float().squeeze(1),
                    sample_rate=sample_rate,
                    n_fft=shape["n_fft"],
                    hop=shape["hop"],
                    window_size=shape["window"],
                    n_mels=shape["n_mels"],
                )
                frames = min(original_mel.shape[-1], generated_mel.shape[-1])
                mel_loss = F.l1_loss(original_mel[..., :frames].float(), generated_mel[..., :frames]) * 45.0
                latent, prior_latent, prior_mean, prior_logs, _, posterior_logs = latent_values
                divergence = kl_loss(prior_latent, posterior_logs, prior_mean, prior_logs, spec_mask)
                adversarial = generator_loss(fake_scores)
                feature_loss = feature_matching_loss(real_maps, fake_maps)
                loss_g = mel_loss + divergence + adversarial + feature_loss
            optimizer_g.zero_grad(set_to_none=True)
            scaler.scale(loss_g).backward()
            scaler.unscale_(optimizer_g)
            torch.nn.utils.clip_grad_norm_(generator.parameters(), 100.0)
            scaler.step(optimizer_g)
            scaler.update()

            global_step += 1
            running += float(loss_g.detach().item())
            progress = 0.35 + 0.58 * ((epoch - 1 + batch_index / len(loader)) / epochs)
            if batch_index == 1 or batch_index == len(loader) or global_step % 10 == 0:
                stage(
                    "training",
                    progress,
                    f"训练轮次 {epoch}/{epochs}，批次 {batch_index}/{len(loader)}",
                    epoch=epoch,
                    step=global_step,
                    loss=round(running / batch_index, 5),
                )
        scheduler_g.step()
        scheduler_d.step()
        if epoch % int(config["save_every"]) == 0 or epoch == epochs:
            torch.save(
                {"model": generator.state_dict(), "optimizer": optimizer_g.state_dict(), "epoch": epoch},
                checkpoint_dir / "generator_latest.pth",
            )
            torch.save(
                {"model": discriminator.state_dict(), "optimizer": optimizer_d.state_dict(), "epoch": epoch},
                checkpoint_dir / "discriminator_latest.pth",
            )
    return generator, args


def export_outputs(config: dict, generator: torch.nn.Module, args: list, speaker_info: list[dict]):
    files_dir = Path(config["files_dir"])
    stem = safe_name(config["name"])
    model_path = files_dir / f"{stem}_{config['id'][:8]}.pth"
    weights = {
        key: value.detach().half().cpu()
        for key, value in generator.state_dict().items()
        if not key.startswith("enc_q.")
    }
    checkpoint = {
        "weight": weights,
        "config": args,
        "info": f"Native streaming trainer, {config['epochs']} epochs",
        "sr": f"{int(config['sample_rate']) // 1000}k",
        "f0": 1,
        "version": "v2",
        "speaker_info": speaker_info,
    }
    torch.save(checkpoint, model_path)

    stage("index", 0.95, "正在建立每位说话人的特征检索索引")
    import faiss
    speaker_outputs = []
    for speaker in speaker_info:
        speaker_id = int(speaker["id"])
        feature_path = Path(config["work_dir"]) / f"index_features_spkid{speaker_id}.npy"
        if not feature_path.is_file():
            continue
        features = np.load(feature_path, mmap_mode="r")
        if features.ndim != 2 or features.shape[0] < 1:
            continue
        vectors = np.asarray(features, dtype=np.float32)
        if vectors.shape[0] > 250000:
            rng = np.random.default_rng(1234 + speaker_id)
            vectors = vectors[rng.choice(vectors.shape[0], 250000, replace=False)]
        index = faiss.IndexFlatL2(vectors.shape[1])
        index.add(vectors)
        index_path = files_dir / f"{stem}_{config['id'][:8]}_spkid{speaker_id}.index"
        faiss.write_index(index, str(index_path))
        speaker_outputs.append(
            {"id": speaker_id, "name": speaker["name"], "index_file": index_path.name}
        )
    if not speaker_outputs:
        raise ValueError("没有可用于索引的特征")
    emit(
        state="completed",
        stage="completed",
        progress=1.0,
        message="训练与索引建立完成",
        model_file=model_path.name,
        index_file=speaker_outputs[0]["index_file"],
        speaker_outputs=speaker_outputs,
        speaker_count=len(speaker_info),
    )


def main() -> int:
    if len(sys.argv) != 2:
        raise SystemExit("usage: training_worker.py JOB_CONFIG")
    with open(sys.argv[1], "r", encoding="utf-8") as handle:
        config = json.load(handle)
    torch.manual_seed(1234)
    np.random.seed(1234)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    shape = model_shape(int(config["sample_rate"]))
    stage("starting", 0.01, f"训练设备：{device}")
    examples, speaker_info = prepare_examples(config, shape, device)
    stage("training", 0.35, f"已准备 {len(examples)} 个训练片段")
    generator, args = train(config, shape, examples, speaker_info, device)
    export_outputs(config, generator, args, speaker_info)
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except SystemExit:
        raise
    except Exception as error:
        emit(state="failed", stage="failed", message="训练失败", error=f"{type(error).__name__}: {error}")
        raise
