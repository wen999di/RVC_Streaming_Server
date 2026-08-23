import ast
import os
"""Minimal Wav2Vec2.0 / HuBERT implementation compatible with fairseq checkpoints.

Replaces the fairseq dependency (~500MB+) with ~250 lines of pure PyTorch.
Only implements the subset of the fairseq API that RVC uses:
  - model.extract_features(source, padding_mask, output_layer)
  - model.final_proj
"""

import pickle as _real_pickle
from typing import Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


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


class _Fp32GroupNorm(nn.GroupNorm):
    """GroupNorm that casts to float32 internally, matching fairseq's Fp32GroupNorm.
    Critical for numerical stability when the model runs in float16 mode."""

    def forward(self, input):
        output = F.group_norm(
            input.float(),
            self.num_groups,
            self.weight.float() if self.weight is not None else None,
            self.bias.float() if self.bias is not None else None,
            self.eps,
        )
        return output.type_as(input)


def _make_conv(in_dim: int, out_dim: int, kernel: int, stride: int,
               with_group_norm: bool = False) -> nn.Sequential:
    """Return a Sequential that mirrors fairseq's internal conv-block layout.

    Fairseq structure (indices in Sequential):
      0 = Conv1d (bias=False), 1 = Dropout(0.0),
      2 = GroupNorm (if with_group_norm) else GELU,
      3 = GELU (only when GroupNorm present at index 2).
    """
    # fairseq uses NO padding on conv layers (see wav2vec2.py line 866).
    layers: list = [
        nn.Conv1d(in_dim, out_dim, kernel, stride=stride, bias=False),
        nn.Dropout(0.0),
    ]
    if with_group_norm:
        # Fp32GroupNorm(dim, dim): casts to float32 internally for stability.
        # Use elementwise_affine=True (default) to have weight+bias params.
        layers.append(_Fp32GroupNorm(num_groups=out_dim, num_channels=out_dim))
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
    """Fairseq-compatible MHA that delegates to PyTorch's fused CUDA kernel
    (F.multi_head_attention_forward), matching fairseq's own MultiheadAttention
    exactly in both computation and performance."""

    def __init__(self, embed_dim: int, num_heads: int, dropout: float = 0.0):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.dropout = dropout

        self.q_proj = nn.Linear(embed_dim, embed_dim)
        self.k_proj = nn.Linear(embed_dim, embed_dim)
        self.v_proj = nn.Linear(embed_dim, embed_dim)
        self.out_proj = nn.Linear(embed_dim, embed_dim)

    def forward(self, x: torch.Tensor,
                key_padding_mask: Optional[torch.Tensor] = None):
        """x: (T, B, D) — same convention as fairseq."""
        return F.multi_head_attention_forward(
            query=x, key=x, value=x,
            embed_dim_to_check=self.embed_dim,
            num_heads=self.num_heads,
            in_proj_weight=torch.empty([0]),
            in_proj_bias=torch.cat((self.q_proj.bias, self.k_proj.bias, self.v_proj.bias)),
            bias_k=None, bias_v=None,
            add_zero_attn=False,
            dropout_p=self.dropout,
            out_proj_weight=self.out_proj.weight,
            out_proj_bias=self.out_proj.bias,
            training=self.training,
            key_padding_mask=key_padding_mask,
            need_weights=False,
            attn_mask=None,
            use_separate_proj_weight=True,
            q_proj_weight=self.q_proj.weight,
            k_proj_weight=self.k_proj.weight,
            v_proj_weight=self.v_proj.weight,
        )[0]


# ---------------------------------------------------------------------------
# Transformer encoder
# ---------------------------------------------------------------------------

class _TransformerEncoderLayer(nn.Module):
    """Matches fairseq TransformerSentenceEncoderLayer: POST-norm + GELU."""

    def __init__(
        self,
        embed_dim: int,
        ffn_dim: int,
        num_heads: int,
        dropout: float = 0.0,
        attention_dropout: float = 0.0,
        activation_dropout: float = 0.0,
        layer_norm_first: bool = False,
    ):
        super().__init__()
        self.self_attn = _MultiheadAttention(embed_dim, num_heads, attention_dropout)
        self.self_attn_layer_norm = nn.LayerNorm(embed_dim)
        self.fc1 = nn.Linear(embed_dim, ffn_dim)
        self.fc2 = nn.Linear(ffn_dim, embed_dim)
        self.final_layer_norm = nn.LayerNorm(embed_dim)
        self.layer_norm_first = layer_norm_first
        self.dropout = dropout
        self.activation_dropout = activation_dropout

    def forward(self, x: torch.Tensor,
                key_padding_mask: Optional[torch.Tensor] = None):
        residual = x

        if self.layer_norm_first:
            x = self.self_attn_layer_norm(x)
            x = self.self_attn(x, key_padding_mask=key_padding_mask)
            x = F.dropout(x, p=self.dropout, training=self.training)
            x = residual + x

            residual = x
            x = self.final_layer_norm(x)
            x = F.gelu(self.fc1(x))
            x = F.dropout(x, p=self.activation_dropout, training=self.training)
            x = self.fc2(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
            x = residual + x
        else:
            # POST-norm: attention → add → norm → FFN → add → norm
            x = self.self_attn(x, key_padding_mask=key_padding_mask)
            x = F.dropout(x, p=self.dropout, training=self.training)
            x = residual + x
            x = self.self_attn_layer_norm(x)

            residual = x
            x = F.gelu(self.fc1(x))
            x = F.dropout(x, p=self.activation_dropout, training=self.training)
            x = self.fc2(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
            x = residual + x
            x = self.final_layer_norm(x)
        return x


class _PositionalConv(nn.Module):
    """Fairseq's make_conv_pos: weight_norm(Conv1d, dim), SamePad, GELU.

    Different fairseq versions use different weight_norm `dim` values
    (dim=2 in current GitHub source, dim=(0,1) in older checkpoints).
    This reads weight_g shape directly from the checkpoint to determine
    which dims were collapsed, so every variant works automatically.

    Also replicates SamePad: for even kernel_size, Conv1d with same-padding
    produces one extra timestep; SamePad strips it so x + pos_conv(x) works.
    """

    def __init__(self, embed_dim, kernel_size, groups, wg_shape):
        super().__init__()
        self.groups = groups
        self.padding = kernel_size // 2
        self.remove = 1 if kernel_size % 2 == 0 else 0  # SamePad logic
        in_ch = embed_dim // groups
        self.weight_v = nn.Parameter(torch.randn(embed_dim, in_ch, kernel_size))
        self.weight_g = nn.Parameter(torch.zeros(wg_shape))
        self.bias = nn.Parameter(torch.zeros(embed_dim))
        # dims where weight_g is 1 are the dims collapsed by the norm
        self._norm_dims = tuple(i for i, s in enumerate(wg_shape) if s == 1)

    def forward(self, x):
        norm = torch.linalg.norm(self.weight_v, dim=self._norm_dims, keepdim=True)
        w = self.weight_g * (self.weight_v / norm)
        out = F.conv1d(x, w, self.bias, stride=1,
                       padding=self.padding, groups=self.groups)
        if self.remove > 0:
            out = out[:, :, :-self.remove]
        return out


class _TransformerEncoder(nn.Module):
    """Wav2Vec2.0 encoder with convolutional positional embedding."""

    def __init__(self, num_layers: int, embed_dim: int, ffn_dim: int,
                 num_heads: int, dropout: float = 0.0,
                 attention_dropout: float = 0.0,
                 activation_dropout: float = 0.0,
                 layer_norm_first: bool = False,
                 layerdrop: float = 0.0,
                 required_seq_len_multiple: int = 1,
                 pos_conv_groups: int | None = None,
                 pos_conv_wg_shape: tuple = (1, 1, 128)):
        super().__init__()
        g = pos_conv_groups if pos_conv_groups is not None else embed_dim
        _conv = _PositionalConv(embed_dim, kernel_size=128, groups=g,
                                wg_shape=pos_conv_wg_shape)
        self.pos_conv = nn.Sequential(_conv, nn.GELU())

        self.dropout = dropout
        self.layer_norm_first = layer_norm_first
        self.layerdrop = float(layerdrop)
        self.required_seq_len_multiple = max(1, int(required_seq_len_multiple))
        self.layer_norm = nn.LayerNorm(embed_dim)
        self.layers = nn.ModuleList([
            _TransformerEncoderLayer(
                embed_dim,
                ffn_dim,
                num_heads,
                dropout=dropout,
                attention_dropout=attention_dropout,
                activation_dropout=activation_dropout,
                layer_norm_first=layer_norm_first,
            )
            for _ in range(num_layers)
        ])

    def forward(self, x, key_padding_mask=None, tgt_layer: Optional[int] = None):
        # x: (T, B, D)
        x_conv = x.permute(1, 2, 0)           # (B, D, T)
        x_conv = self.pos_conv(x_conv)         # (B, D, T)
        x = x + x_conv.permute(2, 0, 1)        # add positional encoding

        if not self.layer_norm_first:
            x = self.layer_norm(x)

        T, B, D = x.shape
        pad_len = (-T) % self.required_seq_len_multiple
        if pad_len > 0:
            x = torch.cat((x, x.new_zeros((pad_len, B, D))), dim=0)
            if key_padding_mask is None:
                key_padding_mask = torch.zeros((B, T), dtype=torch.bool, device=x.device)
            key_padding_mask = torch.cat(
                (
                    key_padding_mask,
                    torch.ones((B, pad_len), dtype=torch.bool, device=key_padding_mask.device),
                ),
                dim=1,
            )

        x = F.dropout(x, p=self.dropout, training=self.training)

        for i, layer in enumerate(self.layers):
            dropout_probability = np.random.random() if self.layerdrop > 0.0 else 1.0
            if (not self.training) or (dropout_probability > self.layerdrop):
                x = layer(x, key_padding_mask)
            if tgt_layer is not None and i == tgt_layer:
                break

        if self.layer_norm_first and tgt_layer is None:
            x = self.layer_norm(x)

        if pad_len > 0:
            x = x[:-pad_len]
            if key_padding_mask is not None:
                key_padding_mask = key_padding_mask[:, :-pad_len]
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
        encoder_attention_dropout: float = 0.0,
        encoder_activation_dropout: float = 0.0,
        layer_norm_first: bool = False,
        encoder_layerdrop: float = 0.0,
        required_seq_len_multiple: int = 1,
        dropout_input: float = 0.0,
        dropout_features: float = 0.0,
        final_dim: int = 256,
        pos_conv_groups: int | None = None,
        pos_conv_wg_shape: tuple = (1, 1, 128),
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
        self.dropout_input = nn.Dropout(dropout_input)
        self.dropout_features = nn.Dropout(dropout_features)

        self.encoder = _TransformerEncoder(
            encoder_layers, encoder_embed_dim, encoder_ffn_embed_dim,
            encoder_attention_heads, encoder_dropout,
            attention_dropout=encoder_attention_dropout,
            activation_dropout=encoder_activation_dropout,
            layer_norm_first=layer_norm_first,
            layerdrop=encoder_layerdrop,
            required_seq_len_multiple=required_seq_len_multiple,
            pos_conv_groups=pos_conv_groups,
            pos_conv_wg_shape=pos_conv_wg_shape,
        )
        self.final_proj = nn.Linear(encoder_embed_dim, final_dim)

        self._stride = 1
        for k, s in _CONV_CONFIG:
            self._stride *= s

    def _compute_conv_output_len(self, input_len: int) -> int:
        # fairseq formula: floor((L - k) / s + 1) — no padding
        L = input_len
        for k, s in _CONV_CONFIG:
            L = (L - k) // s + 1
        return max(L, 0)

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
        features = self.dropout_input(features)

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

        tgt_layer = None if output_layer is None else output_layer - 1
        x = self.encoder(x, key_padding_mask=key_padding_mask, tgt_layer=tgt_layer)

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


_FairseqDictionary = type(
    "Dictionary",
    (),
    {
        "__module__": "fairseq.data.dictionary",
        "__init__": lambda self, *args, **kwargs: None,
        "__setstate__": lambda self, state: self.__dict__.update(state)
        if isinstance(state, dict) else None,
    },
)


def load_hubert(checkpoint_path: str, device: torch.device,
                is_half: bool) -> HubertModel:
    try:
        # Official fairseq HuBERT checkpoints pickle Dictionary metadata even
        # though inference only needs their tensor state. Map that one known
        # type to an inert compatibility object and keep weights-only loading.
        with torch.serialization.safe_globals([_FairseqDictionary]):
            cpt = torch.load(
                checkpoint_path,
                map_location="cpu",
                weights_only=True,
            )
    except Exception as safe_error:
        # Legacy fairseq HuBERT checkpoints may contain pickled config objects.
        # Unsafe pickle is disabled by default; administrators may explicitly
        # opt in only for a checkpoint they have independently verified.
        if os.environ.get("RVC_ALLOW_LEGACY_HUBERT_PICKLE", "").strip() != "1":
            raise RuntimeError(
                "HuBERT checkpoint requires legacy pickle. Set "
                "RVC_ALLOW_LEGACY_HUBERT_PICKLE=1 only for a trusted checkpoint."
            ) from safe_error
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
    if isinstance(conv_feature_layers, str):
        # fairseq configs store conv layers as a string expression
        try:
            conv_feature_layers = ast.literal_eval(conv_feature_layers)
        except Exception:
            conv_feature_layers = None
    if isinstance(conv_feature_layers, list) and conv_feature_layers:
        conv_dim = conv_feature_layers[0][0]
    else:
        conv_dim = 512

    encoder_embed_dim = int(cfg.get("encoder_embed_dim", 768))
    encoder_ffn_embed_dim = int(cfg.get("encoder_ffn_embed_dim", 3072))
    encoder_layers = int(cfg.get("encoder_layers", 12))
    encoder_attention_heads = int(cfg.get("encoder_attention_heads", 12))
    encoder_dropout = float(cfg.get("dropout", 0.0))
    encoder_attention_dropout = float(cfg.get("attention_dropout", 0.0))
    encoder_activation_dropout = float(cfg.get("activation_dropout", 0.0))
    layer_norm_first = bool(cfg.get("layer_norm_first", False))
    encoder_layerdrop = float(cfg.get("encoder_layerdrop", 0.0))
    required_seq_len_multiple = int(cfg.get("required_seq_len_multiple", 1))
    dropout_input = float(cfg.get("dropout_input", 0.0))
    dropout_features = float(cfg.get("dropout_features", 0.0))
    final_dim = int(cfg.get("final_dim", 256))
    if final_dim <= 0:
        final_dim = encoder_embed_dim  # fairseq: final_dim <= 0 → encoder_embed_dim

    # Infer pos_conv configuration from checkpoint parameter shapes.
    # weight_v shape = (embed_dim, embed_dim//groups, kernel) → gives groups.
    # weight_g shape varies with weight_norm's `dim` parameter; we read it
    # directly so _PositionalConv can match the norm behaviour exactly.
    wv = state_dict.get("encoder.pos_conv.0.weight_v")
    wg = state_dict.get("encoder.pos_conv.0.weight_g")
    pos_conv_groups = encoder_embed_dim
    pos_conv_wg_shape = (1, 1, 128)  # fallback
    if wv is not None and wv.ndim >= 3 and wv.shape[1] > 0:
        pos_conv_groups = encoder_embed_dim // wv.shape[1]
    if wg is not None:
        pos_conv_wg_shape = tuple(wg.shape)

    model = HubertModel(
        conv_dim=conv_dim,
        encoder_embed_dim=encoder_embed_dim,
        encoder_ffn_embed_dim=encoder_ffn_embed_dim,
        encoder_layers=encoder_layers,
        encoder_attention_heads=encoder_attention_heads,
        encoder_dropout=encoder_dropout,
        encoder_attention_dropout=encoder_attention_dropout,
        encoder_activation_dropout=encoder_activation_dropout,
        layer_norm_first=layer_norm_first,
        encoder_layerdrop=encoder_layerdrop,
        required_seq_len_multiple=required_seq_len_multiple,
        dropout_input=dropout_input,
        dropout_features=dropout_features,
        final_dim=final_dim,
        pos_conv_groups=pos_conv_groups,
        pos_conv_wg_shape=pos_conv_wg_shape,
    )

    # strict=False: checkpoint may contain pre-training heads (mask_emb,
    # label_embs_concat, quantizer.*) that we don't need for inference.
    model.load_state_dict(state_dict, strict=False)
    model = model.to(device)
    model = model.half() if is_half else model.float()
    model.eval()
    return model
