from __future__ import annotations

import logging
import os
import threading
import time
from collections import OrderedDict

import torch


logger = logging.getLogger(__name__)
_probe_lock = threading.Lock()
_probe_results: dict[str, bool] = {}


def _probe(device: torch.device) -> bool:
    key = str(device)
    with _probe_lock:
        if key in _probe_results:
            return _probe_results[key]
        try:
            parsed = device
            if parsed.index is None:
                parsed = torch.device("cuda", torch.cuda.current_device())
            with torch.cuda.device(parsed):
                source = torch.arange(32, device=parsed, dtype=torch.float32)
                graph = torch.cuda.CUDAGraph()
                with torch.cuda.graph(graph):
                    captured = source.square() + 1.0
                source.copy_(torch.arange(32, device=parsed, dtype=torch.float32))
                graph.replay()
                torch.cuda.synchronize(parsed)
                expected = torch.arange(32, dtype=torch.float32).square() + 1.0
                supported = bool(torch.equal(captured.cpu(), expected))
        except Exception as error:
            logger.warning("CUDA Graph support probe failed on %s: %s", device, error)
            supported = False
        _probe_results[key] = supported
        return supported


def enabled(device) -> bool:
    setting = os.environ.get("RVC_CUDA_GRAPH", "auto").strip().lower()
    if setting in {"0", "false", "off", "no"}:
        return False
    try:
        parsed = device if isinstance(device, torch.device) else torch.device(device)
    except Exception:
        return False
    available = (
        parsed.type == "cuda"
        and torch.cuda.is_available()
        and hasattr(torch.cuda, "CUDAGraph")
        and hasattr(torch.cuda, "graph")
    )
    return bool(available and _probe(parsed))


def _signature(tensor: torch.Tensor) -> tuple:
    return (
        tuple(tensor.shape), tuple(tensor.stride()), str(tensor.dtype), str(tensor.device),
        bool(tensor.requires_grad),
    )


def _clone(value):
    if torch.is_tensor(value):
        return value.clone()
    if isinstance(value, tuple):
        return tuple(_clone(item) for item in value)
    if isinstance(value, list):
        return [_clone(item) for item in value]
    if isinstance(value, dict):
        return {key: _clone(item) for key, item in value.items()}
    return value


class _Captured:
    def __init__(self, function, inputs: tuple[torch.Tensor, ...]):
        started = time.perf_counter()
        self.lock = threading.RLock()
        self.inputs = tuple(torch.empty_like(value) for value in inputs)
        for target, value in zip(self.inputs, inputs):
            target.copy_(value)
        device = self.inputs[0].device
        current = torch.cuda.current_stream(device)
        warmup = torch.cuda.Stream(device=device)
        warmup.wait_stream(current)
        with torch.cuda.stream(warmup), torch.inference_mode():
            for _ in range(3):
                output = function(*self.inputs)
        current.wait_stream(warmup)
        torch.cuda.synchronize(device)
        self.graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(self.graph), torch.inference_mode():
            self.output = function(*self.inputs)
        self.done_event = None
        del output
        self.capture_ms = (time.perf_counter() - started) * 1000.0

    def replay(self, inputs: tuple[torch.Tensor, ...]):
        with self.lock:
            stream = torch.cuda.current_stream(self.inputs[0].device)
            if self.done_event is not None:
                stream.wait_event(self.done_event)
            for target, value in zip(self.inputs, inputs):
                target.copy_(value, non_blocking=True)
            self.graph.replay()
            output = _clone(self.output)
            self.done_event = torch.cuda.Event(blocking=False)
            self.done_event.record(stream)
            return output


class GraphCache:
    def __init__(self) -> None:
        self.entries = OrderedDict()
        self.failures: set[tuple] = set()
        self.lock = threading.RLock()
        self.captures = 0
        self.replays = 0
        self.fallbacks = 0
        self.evictions = 0
        self.capture_ms = 0.0

    def run(self, namespace: str, function, inputs: tuple[torch.Tensor, ...]):
        key = (str(namespace),) + tuple(_signature(value) for value in inputs)
        with self.lock:
            if key in self.failures:
                self.fallbacks += 1
                return function(*inputs)
            entry = self.entries.get(key)
            if entry is None:
                try:
                    entry = _Captured(function, inputs)
                    self.entries[key] = entry
                    self.captures += 1
                    self.capture_ms += entry.capture_ms
                    maximum = max(1, int(os.environ.get("RVC_CUDA_GRAPH_CACHE_SIZE", "12")))
                    while len(self.entries) > maximum:
                        self.entries.popitem(last=False)
                        self.evictions += 1
                except Exception as error:
                    self.failures.add(key)
                    self.fallbacks += 1
                    logger.warning("CUDA Graph capture failed for %s; using eager: %s", namespace, error)
                    return function(*inputs)
            else:
                self.entries.move_to_end(key)
        output = entry.replay(inputs)
        with self.lock:
            self.replays += 1
        return output

    def stats(self) -> dict:
        with self.lock:
            return {
                "entries": len(self.entries),
                "failures": len(self.failures),
                "captures": self.captures,
                "replays": self.replays,
                "fallbacks": self.fallbacks,
                "evictions": self.evictions,
                "capture_ms": round(self.capture_ms, 3),
            }


def run(owner, namespace: str, function, *inputs: torch.Tensor):
    if not inputs or not enabled(inputs[0].device):
        return function(*inputs)
    cache = getattr(owner, "_streaming_cuda_graphs", None)
    if cache is None:
        cache = GraphCache()
        setattr(owner, "_streaming_cuda_graphs", cache)
    return cache.run(namespace, function, tuple(inputs))


def clear(owner) -> None:
    cache = getattr(owner, "_streaming_cuda_graphs", None)
    if cache is not None:
        cache.entries.clear()
        cache.failures.clear()
        delattr(owner, "_streaming_cuda_graphs")


def stats(owner) -> dict:
    cache = getattr(owner, "_streaming_cuda_graphs", None)
    return cache.stats() if cache is not None else {
        "entries": 0, "failures": 0, "captures": 0, "replays": 0,
        "fallbacks": 0, "evictions": 0, "capture_ms": 0.0,
    }
