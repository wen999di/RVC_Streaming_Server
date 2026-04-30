"""Minimal HuBERT implementation compatible with fairseq checkpoints.

Replaces the fairseq dependency (~500MB+) with ~200 lines of pure PyTorch.
Only implements the subset of the fairseq API that RVC uses:
  - HubertModel.extract_features(source, padding_mask, output_layer)
  - HubertModel.final_proj
"""

import math
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# Conv feature extractor (matches fairseq ConvFeatureExtractionModel)
# ---------------------------------------------------------------------------

def _make_conv_layer(in_dim: int, out_dim: int, kernel: int, stride: int):
    return nn.Sequential(
        nn.Conv1d(in_dim, out_dim, kernel, stride=stride, padding=kernel // 2, bias=False),
        nn.Dropout(0.0),
        nn.GELU(),
    )


def _make_conv_layer_norm(in_dim: int, out_dim: int, kernel: int, stride: int):
    return nn.Sequential(
        nn.Conv1d(in_dim, out_dim, kernel, stride=stride, padding=kernel // 2, bias=False),
        nn.Dropout(0.0),
        nn.LayerNorm(out_dim),
        nn.GELU(),
    )


# Standard hubert_base conv config: kernel, stride
_CONV_CONFIG = [(10, 5), (3, 2), (3, 2), (3, 2), (3, 2), (2, 2), (2, 2)]


def _build_conv_layers(conv_dim: int = 512):
    layers = nn.ModuleList()
    in_dim = 1
    for i, (k, s) in enumerate(_CONV_CONFIG):
        is_last = i == len(_CONV_CONFIG) - 1
        make = _make_conv_layer if is_last else _make_conv_layer_norm
        layers.append(make(in_dim, conv_dim, k, s))
        in_dim = conv_dim
    return layers


# ---------------------------------------------------------------------------
# Multi-head attention (matches fairseq MultiheadAttention)
# ---------------------------------------------------------------------------

class _MultiheadAttention(nn.Module):
    """Fairseq-compatible multi-head attention with separate Q/K/V/O projections."""

    def __init__(self, embed_dim: int, num_heads: int, dropout: float = 0.0):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.scaling = self.head_dim ** -0.5
        self.dropout = dropout

        self.q_proj = nn.Linear(embed_dim, embed_dim)
        self.k_proj = nn.Linear(embed_dim, embed_dim)
        self.v_proj = nn.Linear(embed_dim, embed_dim)
        self.out_proj = nn.Linear(embed_dim, embed_dim)

    def forward(self, x: torch.Tensor, key_padding_mask: Optional[torch.Tensor] = None):
        """x: (T, B, D)"""
        T, B, D = x.shape
        q = self.q_proj(x).view(T, B * self.num_heads, self.head_dim).transpose(0, 1)
        k = self.k_proj(x).view(T, B * self.num_heads, self.head_dim).transpose(0, 1)
        v = self.v_proj(x).view(T, B * self.num_heads, self.head_dim).transpose(0, 1)

        attn_weights = torch.bmm(q, k.transpose(1, 2)) * self.scaling

        if key_padding_mask is not None:
            # key_padding_mask: (B, T)  →  broadcast to (B*H, T, T)
            attn_weights = attn_weights.view(B, self.num_heads, T, T)
            attn_weights = attn_weights.masked_fill(
                key_padding_mask.unsqueeze(1).unsqueeze(2), float("-inf")
            )
            attn_weights = attn_weights.view(B * self.num_heads, T, T)

        attn_weights = F.softmax(attn_weights, dim=-1, dtype=torch.float32).to(q.dtype)
        attn = torch.bmm(attn_weights, v)
        attn = attn.transpose(0, 1).contiguous().view(T, B, D)
        attn = self.out_proj(attn)
        return attn


# ---------------------------------------------------------------------------
# Transformer encoder (matches fairseq TransformerEncoder)
# ---------------------------------------------------------------------------

class _TransformerEncoderLayer(nn.Module):
    def __init__(self, embed_dim: int, ffn_dim: int, num_heads: int, dropout: float = 0.0):
        super().__init__()
        self.self_attn = _MultiheadAttention(embed_dim, num_heads, dropout)
        self.self_attn_layer_norm = nn.LayerNorm(embed_dim)
        self.fc1 = nn.Linear(embed_dim, ffn_dim)
        self.fc2 = nn.Linear(ffn_dim, embed_dim)
        self.final_layer_norm = nn.LayerNorm(embed_dim)
        self.dropout = dropout

    def forward(self, x: torch.Tensor, key_padding_mask: Optional[torch.Tensor] = None):
        residual = x
        x = self.self_attn_layer_norm(x)
        x = self.self_attn(x, key_padding_mask=key_padding_mask)
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = residual + x

        residual = x
        x = self.final_layer_norm(x)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = residual + x
        return x


class _TransformerEncoder(nn.Module):
    def __init__(self, num_layers: int, embed_dim: int, ffn_dim: int, num_heads: int, dropout: float = 0.0):
        super().__init__()
        self.layers = nn.ModuleList([
            _TransformerEncoderLayer(embed_dim, ffn_dim, num_heads, dropout)
            for _ in range(num_layers)
        ])
        self.layer_norm = nn.Identity()  # hubert_base doesn't use final layer norm

    def forward(self, x, key_padding_mask=None):
        for layer in self.layers:
            x = layer(x, key_padding_mask)
        return x


# ---------------------------------------------------------------------------
# HuBERT Model
# ---------------------------------------------------------------------------

class HubertModel(nn.Module):
    def __init__(
        self,
        conv_dim: int = 512,
        encoder_embed_dim: int = 768,
        encoder_ffn_embed_dim: int = 3072,
        encoder_layers: int = 12,
        encoder_attention_heads: int = 12,
        encoder_dropout: float = 0.0,
        final_dim: int = 256,
    ):
        super().__init__()
        self.feature_extractor = nn.Module()
        self.feature_extractor.conv_layers = _build_conv_layers(conv_dim)

        proj_in = conv_dim
        proj_out = encoder_embed_dim
        self.post_extract_proj = (
            nn.Linear(proj_in, proj_out) if proj_in != proj_out else nn.Identity()
        )

        self.layer_norm = nn.LayerNorm(encoder_embed_dim)
        self.encoder = _TransformerEncoder(
            encoder_layers, encoder_embed_dim, encoder_ffn_embed_dim,
            encoder_attention_heads, encoder_dropout,
        )
        self.final_proj = nn.Linear(encoder_embed_dim, final_dim)

        self._stride = 1
        for k, s in _CONV_CONFIG:
            self._stride *= s

    def _compute_conv_output_len(self, input_len: int) -> int:
        L = input_len
        for k, s in _CONV_CONFIG:
            p = k // 2
            L = (L + 2 * p - k) // s + 1
        return L

    def extract_features(
        self,
        source: torch.Tensor,
        padding_mask: Optional[torch.Tensor] = None,
        output_layer: Optional[int] = None,
    ):
        # source: (B, T) raw waveform
        features = source.unsqueeze(1)  # (B, 1, T)
        for conv in self.feature_extractor.conv_layers:
            features = conv(features)
        features = features.transpose(1, 2)  # (B, T', C)
        features = self.layer_norm(features)
        features = self.post_extract_proj(features)

        # Compute key padding mask for transformer
        if padding_mask is not None and padding_mask.any():
            input_lengths = (padding_mask.logical_not()).sum(dim=1)
            conv_lengths = torch.tensor(
                [self._compute_conv_output_len(int(l.item())) for l in input_lengths],
                device=input_lengths.device,
            )
            max_len = features.shape[1]
            key_padding_mask = torch.arange(max_len, device=features.device).unsqueeze(0) >= conv_lengths.unsqueeze(1)
        else:
            key_padding_mask = None

        x = features.transpose(0, 1)  # (T', B, C)
        for i, layer in enumerate(self.encoder.layers):
            x = layer(x, key_padding_mask=key_padding_mask)
            if output_layer is not None and i + 1 == output_layer:
                x = x.transpose(0, 1)  # back to (B, T', C)
                return [x]

        x = x.transpose(0, 1)
        return [x]


# ---------------------------------------------------------------------------
# Safe checkpoint loader (bypasses fairseq pickle references)
# ---------------------------------------------------------------------------

import pickle as _pickle
import sys as _sys


def _patch_pickle_for_fairseq():
    """Make pickle.Unpickler tolerant of missing fairseq/omegaconf modules.

    fairseq checkpoints contain pickled fairseq config objects. torch.load
    with weights_only=False tries to reconstruct them via pickle, which requires
    those modules to be importable. This patch returns inert dummy objects for
    any unrecognised module, allowing torch.load to succeed.
    """
    _original_find_class = _pickle.Unpickler.find_class

    def _dummy_class():
        return type("_Dummy", (), {
            "__init__": lambda self, *a, **kw: None,
            "__setstate__": lambda self, state: None,
        })

    def _patched_find_class(self, module, name):
        if module.startswith(("fairseq", "omegaconf")):
            return _dummy_class()
        try:
            return _original_find_class(self, module, name)
        except (ModuleNotFoundError, ImportError):
            return _dummy_class()

    _pickle.Unpickler.find_class = _patched_find_class
    return _original_find_class


def _restore_pickle_find_class(original):
    _pickle.Unpickler.find_class = original


def _ensure_dummy_modules():
    """Register dummy modules so import statements inside pickle don't crash."""
    for mod in ("fairseq", "fairseq.models", "fairseq.modules",
                "fairseq.checkpoint_utils", "fairseq.data", "fairseq.tasks",
                "fairseq.dataclass", "fairseq_cli", "omegaconf"):
        if mod not in _sys.modules:
            _sys.modules[mod] = type(_sys)(mod)


def load_hubert(checkpoint_path: str, device: torch.device, is_half: bool) -> HubertModel:
    _ensure_dummy_modules()
    _orig_find = _patch_pickle_for_fairseq()
    try:
        cpt = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    finally:
        _restore_pickle_find_class(_orig_find)

    if not isinstance(cpt, dict):
        raise RuntimeError(f"Unrecognised checkpoint format: {type(cpt)}")

    cfg = cpt.get("cfg", {})
    if isinstance(cfg, dict):
        cfg = cfg.get("model", {}) if isinstance(cfg.get("model"), dict) else {}
    else:
        cfg = {}
    state_dict = cpt.get("model", cpt)
    if not isinstance(state_dict, dict):
        raise RuntimeError(f"Checkpoint has no state dict under 'model' key")

    conv_dim = cfg.get("conv_feature_layers", "[(512,10,5)] + [(512,3,2)] * 4 + [(512,2,2)] * 2")
    if isinstance(conv_dim, str):
        conv_dim = 512
    elif isinstance(conv_dim, list):
        conv_dim = conv_dim[0][0] if conv_dim else 512
    else:
        conv_dim = 512

    encoder_embed_dim = int(cfg.get("encoder_embed_dim", 768))
    encoder_ffn_embed_dim = int(cfg.get("encoder_ffn_embed_dim", 3072))
    encoder_layers = int(cfg.get("encoder_layers", 12))
    encoder_attention_heads = int(cfg.get("encoder_attention_heads", 12))
    encoder_dropout = float(cfg.get("dropout", 0.0))
    final_dim = int(cfg.get("final_dim", 256))

    model = HubertModel(
        conv_dim=conv_dim,
        encoder_embed_dim=encoder_embed_dim,
        encoder_ffn_embed_dim=encoder_ffn_embed_dim,
        encoder_layers=encoder_layers,
        encoder_attention_heads=encoder_attention_heads,
        encoder_dropout=encoder_dropout,
        final_dim=final_dim,
    )

    model.load_state_dict(state_dict, strict=True)
    model = model.to(device)
    model = model.half() if is_half else model.float()
    model.eval()
    return model
