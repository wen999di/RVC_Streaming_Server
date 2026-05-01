"""Minimal Wav2Vec2.0 / HuBERT implementation compatible with fairseq checkpoints.

Replaces the fairseq dependency (~500MB+) with ~250 lines of pure PyTorch.
Only implements the subset of the fairseq API that RVC uses:
  - model.extract_features(source, padding_mask, output_layer)
  - model.final_proj
"""

import pickle as _real_pickle
import sys as _sys
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils import weight_norm


# ---------------------------------------------------------------------------
# Conv feature extractor  (matches fairseq ConvFeatureExtractionModel)
# ---------------------------------------------------------------------------
# fairseq Wav2Vec2.0 base in "default" extractor_mode:
#   Layer 0: Conv1d -> Dropout -> GroupNorm -> GELU
#   Layers 1-6: Conv1d -> Dropout -> GELU
# Fairseq's Fp32GroupNorm(dim, dim) = GroupNorm(num_groups=dim, num_channels=dim),
# which normalises each channel independently (equivalent to InstanceNorm1d).
# PyTorch 2.x GroupNorm already computes in fp32 internally when input is fp16.

_CONV_CONFIG = [
    # kernel, stride
    (10, 5),   # layer 0 — has GroupNorm
    (3, 2),    # layer 1
    (3, 2),    # layer 2
    (3, 2),    # layer 3
    (3, 2),    # layer 4
    (2, 2),    # layer 5
    (2, 2),    # layer 6
]


def _make_conv(in_dim: int, out_dim: int, kernel: int, stride: int,
               with_group_norm: bool = False) -> nn.Sequential:
    """Return a Sequential that mirrors fairseq's internal conv-block layout.

    Fairseq structure (indices in Sequential):
      0 = Conv1d (bias=False), 1 = Dropout(0.0),
      2 = GroupNorm (if with_group_norm) else GELU,
      3 = GELU (only when GroupNorm present at index 2).
    """
    layers: list = [
        nn.Conv1d(in_dim, out_dim, kernel, stride=stride,
                  padding=kernel // 2, bias=False),
        nn.Dropout(0.0),
    ]
    if with_group_norm:
        # Fp32GroupNorm(dim, dim) – one group per channel = InstanceNorm1d
        layers.append(nn.GroupNorm(num_groups=out_dim, num_channels=out_dim))
    layers.append(nn.GELU())
    return nn.Sequential(*layers)


def _build_conv_layers(conv_dim: int = 512) -> nn.ModuleList:
    layers = nn.ModuleList()
    in_dim = 1
    for i, (k, s) in enumerate(_CONV_CONFIG):
        layers.append(_make_conv(in_dim, conv_dim, k, s, with_group_norm=(i == 0)))
        in_dim = conv_dim
    return layers


# ---------------------------------------------------------------------------
# Multi-head attention  (fairseq-compatible Q/K/V/O projections)
# ---------------------------------------------------------------------------

class _MultiheadAttention(nn.Module):
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

    def forward(self, x: torch.Tensor,
                key_padding_mask: Optional[torch.Tensor] = None):
        """x: (T, B, D)"""
        T, B, D = x.shape
        q = self.q_proj(x).view(T, B * self.num_heads, self.head_dim).transpose(0, 1)
        k = self.k_proj(x).view(T, B * self.num_heads, self.head_dim).transpose(0, 1)
        v = self.v_proj(x).view(T, B * self.num_heads, self.head_dim).transpose(0, 1)

        attn_weights = torch.bmm(q, k.transpose(1, 2)) * self.scaling

        if key_padding_mask is not None:
            attn_weights = attn_weights.view(B, self.num_heads, T, T)
            attn_weights = attn_weights.masked_fill(
                key_padding_mask.unsqueeze(1).unsqueeze(2), float("-inf"),
            )
            attn_weights = attn_weights.view(B * self.num_heads, T, T)

        attn_weights = F.softmax(attn_weights, dim=-1, dtype=torch.float32).to(q.dtype)
        attn = torch.bmm(attn_weights, v)
        attn = attn.transpose(0, 1).contiguous().view(T, B, D)
        attn = self.out_proj(attn)
        return attn


# ---------------------------------------------------------------------------
# Transformer encoder
# ---------------------------------------------------------------------------

class _TransformerEncoderLayer(nn.Module):
    def __init__(self, embed_dim: int, ffn_dim: int, num_heads: int,
                 dropout: float = 0.0):
        super().__init__()
        self.self_attn = _MultiheadAttention(embed_dim, num_heads, dropout)
        self.self_attn_layer_norm = nn.LayerNorm(embed_dim)
        self.fc1 = nn.Linear(embed_dim, ffn_dim)
        self.fc2 = nn.Linear(ffn_dim, embed_dim)
        self.final_layer_norm = nn.LayerNorm(embed_dim)
        self.dropout = dropout

    def forward(self, x: torch.Tensor,
                key_padding_mask: Optional[torch.Tensor] = None):
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
    """Wav2Vec2.0 encoder with convolutional positional embedding."""

    def __init__(self, num_layers: int, embed_dim: int, ffn_dim: int,
                 num_heads: int, dropout: float = 0.0):
        super().__init__()
        # Convolutional positional encoding (weight_norm applied after init)
        self.pos_conv = nn.Sequential(
            nn.Conv1d(embed_dim, embed_dim, kernel_size=128,
                      groups=16, padding=64),
            nn.GELU(),
        )
        weight_norm(self.pos_conv[0], name="weight")

        self.layer_norm = nn.LayerNorm(embed_dim)
        self.layers = nn.ModuleList([
            _TransformerEncoderLayer(embed_dim, ffn_dim, num_heads, dropout)
            for _ in range(num_layers)
        ])

    def forward(self, x, key_padding_mask=None):
        # x: (T, B, D)
        x_conv = x.permute(1, 2, 0)           # (B, D, T)
        x_conv = self.pos_conv(x_conv)         # (B, D, T)
        x = x + x_conv.permute(2, 0, 1)        # add positional encoding

        x = self.layer_norm(x)

        for layer in self.layers:
            x = layer(x, key_padding_mask)
        return x


# ---------------------------------------------------------------------------
# Wav2Vec2.0 Model  (compatible with fairseq's Wav2Vec2Model / HubertModel)
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

        # LayerNorm is applied to conv output (512-d) BEFORE projection.
        self.layer_norm = nn.LayerNorm(conv_dim)

        proj_in = conv_dim
        proj_out = encoder_embed_dim
        self.post_extract_proj = (
            nn.Linear(proj_in, proj_out) if proj_in != proj_out else nn.Identity()
        )

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
        # --- conv feature extraction ---
        features = source.unsqueeze(1)                 # (B, 1, T)
        for conv in self.feature_extractor.conv_layers:
            features = conv(features)
        features = features.transpose(1, 2)            # (B, T', conv_dim)
        features = self.layer_norm(features)            # (B, T', conv_dim)
        features = self.post_extract_proj(features)     # (B, T', embed_dim)

        # --- compute key padding mask for transformer ---
        if padding_mask is not None and padding_mask.any():
            input_lengths = (padding_mask.logical_not()).sum(dim=1)
            conv_lengths = torch.tensor(
                [self._compute_conv_output_len(int(l.item()))
                 for l in input_lengths],
                device=input_lengths.device,
            )
            max_len = features.shape[1]
            key_padding_mask = torch.arange(
                max_len, device=features.device
            ).unsqueeze(0) >= conv_lengths.unsqueeze(1)
        else:
            key_padding_mask = None

        # --- encoder (pos_conv + layer_norm + transformer layers) ---
        x = features.transpose(0, 1)                   # (T', B, embed_dim)

        x_conv = x.permute(1, 2, 0)                    # (B, embed_dim, T')
        x_conv = self.encoder.pos_conv(x_conv)         # (B, embed_dim, T')
        x = x + x_conv.permute(2, 0, 1)                # add positional encoding

        x = self.encoder.layer_norm(x)

        for i, layer in enumerate(self.encoder.layers):
            x = layer(x, key_padding_mask=key_padding_mask)
            if output_layer is not None and i + 1 == output_layer:
                x = x.transpose(0, 1)                  # (B, T', embed_dim)
                return [x]

        x = x.transpose(0, 1)
        return [x]


# ---------------------------------------------------------------------------
# Safe checkpoint loader  (bypasses fairseq pickle references)
# ---------------------------------------------------------------------------

def _dummy_class():
    return type("_Dummy", (), {
        "__init__": lambda self, *a, **kw: None,
        "__setstate__": lambda self, state: None,
    })


class _SafeUnpickler(_real_pickle.Unpickler):
    """Custom Unpickler that returns dummy objects for missing fairseq modules.

    Wav2Vec2.0 / HuBERT checkpoints contain pickled fairseq config objects.
    When fairseq is not installed, standard pickle raises ModuleNotFoundError.
    This subclass returns inert objects so torch.load can still extract the
    tensor state dict.
    """

    def find_class(self, module, name):
        if module.startswith(("fairseq", "omegaconf")):
            return _dummy_class()
        try:
            return super().find_class(module, name)
        except (ModuleNotFoundError, ImportError):
            return _dummy_class()


class _SafePickleModule:
    """Module-like proxy: delegates everything to the real pickle module,
    except Unpickler which uses our safe subclass."""

    Unpickler = _SafeUnpickler

    def __getattr__(self, name):
        return getattr(_real_pickle, name)


def _ensure_dummy_modules():
    """Register fake fairseq / omegaconf modules in sys.modules so that
    'import fairseq' statements inside pickle don't crash."""
    for mod in ("fairseq", "fairseq.models", "fairseq.modules",
                "fairseq.checkpoint_utils", "fairseq.data", "fairseq.tasks",
                "fairseq.dataclass", "fairseq_cli", "omegaconf"):
        if mod not in _sys.modules:
            _sys.modules[mod] = type(_sys)(mod)


def load_hubert(checkpoint_path: str, device: torch.device,
                is_half: bool) -> HubertModel:
    _ensure_dummy_modules()
    cpt = torch.load(
        checkpoint_path,
        map_location="cpu",
        weights_only=False,
        pickle_module=_SafePickleModule(),
    )

    if not isinstance(cpt, dict):
        raise RuntimeError(f"Unrecognised checkpoint format: {type(cpt)}")

    # cfg may be a real dict or a dummy object (from patched pickle)
    cfg = cpt.get("cfg", {})
    if isinstance(cfg, dict):
        cfg = cfg.get("model", {}) if isinstance(cfg.get("model"), dict) else {}
    else:
        cfg = {}

    state_dict = cpt.get("model", cpt)
    if not isinstance(state_dict, dict):
        raise RuntimeError("Checkpoint has no state dict under 'model' key")

    # --- read architecture params from checkpoint config ---
    conv_feature_layers = cfg.get("conv_feature_layers", None)
    if isinstance(conv_feature_layers, list) and conv_feature_layers:
        conv_dim = conv_feature_layers[0][0]
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

    # strict=False: checkpoint may contain pre-training heads (mask_emb,
    # label_embs_concat, quantizer.*) that we don't need for inference.
    model.load_state_dict(state_dict, strict=False)
    model = model.to(device)
    model = model.half() if is_half else model.float()
    model.eval()
    return model
