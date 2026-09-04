from __future__ import annotations

import json
import sys
import traceback
from pathlib import Path

import numpy as np


if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")


def emit(**event) -> None:
    print(json.dumps(event, ensure_ascii=False), flush=True)


def _for_audio_writer(audio: object) -> np.ndarray:
    array = np.asarray(audio, dtype=np.float32)
    if array.ndim == 1:
        return np.ascontiguousarray(array)
    if array.ndim != 2:
        raise ValueError(f"PyMSS 输出维度无效: {array.shape}")
    if array.shape[1] in (1, 2):
        return np.ascontiguousarray(array)
    if array.shape[0] in (1, 2):
        return np.ascontiguousarray(array.T)
    raise ValueError(f"PyMSS 输出声道数无效: {array.shape}")


def run(job_path: Path) -> None:
    with open(job_path, "r", encoding="utf-8") as handle:
        config = json.load(handle)

    from pymss import MSSeparator, load_audio, save_audio

    audio_files = config.get("audio_files")
    if not isinstance(audio_files, list) or not audio_files:
        raise ValueError("没有可供 PyMSS 处理的音频")

    output_dir = Path(config["output_dir"]).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    stem = str(config["stem"])
    device = str(config.get("device") or "auto")
    emit(type="pymss_status", message="正在加载 PyMSS 模型", progress=0.0)
    separator = MSSeparator(
        model_type=str(config["model_type"]),
        model_path=str(config["model_path"]),
        config_path=str(config["config_path"]),
        device=device,
        device_ids=[0],
        output_format="wav",
        use_tta=False,
        store_dirs={},
        inference_params={
            "normalize": False,
            "standardize": False,
            "use_amp": device in {"auto", "cuda"},
        },
    )
    processed: list[dict] = []
    try:
        total = len(audio_files)
        for index, item in enumerate(audio_files):
            source = Path(str(item["path"])).resolve()
            mix, sample_rate = load_audio(str(source), sr=44100, mono=False)
            stems = separator.separate(mix, pbar=False)
            if stem not in stems:
                available = ", ".join(sorted(str(key) for key in stems))
                raise RuntimeError(f"分离模型没有输出 {stem}；可用输出: {available}")
            desired = np.asarray(stems[stem])
            if not np.isfinite(desired).all():
                raise FloatingPointError(f"{source.name} 的分离结果包含 NaN/Inf")
            output = output_dir / f"{index:06d}_{source.stem}_{stem}.wav"
            save_audio(
                str(output),
                _for_audio_writer(desired),
                int(sample_rate),
                "wav",
                {"wav_bit_depth": "FLOAT"},
            )
            if not output.is_file() or output.stat().st_size == 0:
                raise RuntimeError(f"未能写入分离结果: {output.name}")
            processed.append(
                {
                    "name": output.name,
                    "path": str(output),
                    "speaker": str(item["speaker"]),
                }
            )
            emit(
                type="pymss_status",
                message=f"已完成训练前处理 {index + 1}/{total}",
                progress=(index + 1) / total,
            )
    finally:
        close = getattr(separator, "close", None)
        if callable(close):
            close()

    emit(type="pymss_result", audio_files=processed)


def main() -> int:
    if len(sys.argv) != 2:
        emit(type="pymss_error", error="用法: pymss_preprocess_worker.py <job.json>")
        return 2
    try:
        run(Path(sys.argv[1]).resolve())
        return 0
    except Exception as error:
        emit(
            type="pymss_error",
            error=f"{type(error).__name__}: {error}",
            traceback=traceback.format_exc(limit=8),
        )
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
