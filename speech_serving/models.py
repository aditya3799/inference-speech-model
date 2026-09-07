"""
Speech Model Architectures for Paged Dual-Cache Inference.

Provides:
1. StreamingMultiCodebookTTSModel: Multi-codebook autoregressive TTS decoder model.
2. WhisperClassSpeechEncoderDecoderModel: Speech Encoder-Decoder with cross-attention.
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple, List, Dict, Any
from transformers.cache_utils import DynamicCache, DynamicLayer


class MultiCodebookEmbedding(nn.Module):
    """
    Embedding layer for neural audio codecs with K parallel codebooks per timestep.
    Summing embeddings across K codebooks for input representation.
    """

    def __init__(self, num_codebooks: int, vocab_size: int, embed_dim: int):
        super().__init__()
        self.num_codebooks = num_codebooks
        self.vocab_size = vocab_size
        self.embed_dim = embed_dim
        self.embeddings = nn.ModuleList([
            nn.Embedding(vocab_size, embed_dim) for _ in range(num_codebooks)
        ])

    def forward(self, input_codebooks: torch.Tensor) -> torch.Tensor:
        """
        Args:
            input_codebooks: Tensor of shape [batch_size, seq_len, num_codebooks]
        Returns:
            Tensor of shape [batch_size, seq_len, embed_dim]
        """
        batch_size, seq_len, num_cb = input_codebooks.shape
        assert num_cb == self.num_codebooks, f"Expected {self.num_codebooks} codebooks, got {num_cb}"

        embeds = self.embeddings[0](input_codebooks[:, :, 0])
        for k in range(1, self.num_codebooks):
            embeds = embeds + self.embeddings[k](input_codebooks[:, :, k])
        return embeds


class MultiCodebookHeads(nn.Module):
    """
    Multiple linear output heads predicting next-step tokens across K codebooks.
    """

    def __init__(self, num_codebooks: int, hidden_dim: int, vocab_size: int):
        super().__init__()
        self.num_codebooks = num_codebooks
        self.vocab_size = vocab_size
        self.heads = nn.ModuleList([
            nn.Linear(hidden_dim, vocab_size, bias=False) for _ in range(num_codebooks)
        ])

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """
        Args:
            hidden_states: Tensor of shape [batch_size, seq_len, hidden_dim]
        Returns:
            Tensor of shape [batch_size, seq_len, num_codebooks, vocab_size]
        """
        logits_list = [head(hidden_states) for head in self.heads]
        return torch.stack(logits_list, dim=2)


class DecoderLayer(nn.Module):
    """Transformer decoder layer supporting causal self-attention and optional cross-attention."""

    def __init__(self, hidden_dim: int, num_heads: int, head_dim: int, is_decoder: bool = True):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.is_decoder = is_decoder

        self.self_attn_q = nn.Linear(hidden_dim, num_heads * head_dim, bias=False)
        self.self_attn_k = nn.Linear(hidden_dim, num_heads * head_dim, bias=False)
        self.self_attn_v = nn.Linear(hidden_dim, num_heads * head_dim, bias=False)
        self.self_attn_out = nn.Linear(num_heads * head_dim, hidden_dim, bias=False)

        # Cross-attention projections (if encoder-decoder)
        self.cross_attn_q = nn.Linear(hidden_dim, num_heads * head_dim, bias=False)
        self.cross_attn_k = nn.Linear(hidden_dim, num_heads * head_dim, bias=False)
        self.cross_attn_v = nn.Linear(hidden_dim, num_heads * head_dim, bias=False)
        self.cross_attn_out = nn.Linear(num_heads * head_dim, hidden_dim, bias=False)

        self.norm1 = nn.LayerNorm(hidden_dim)
        self.norm2 = nn.LayerNorm(hidden_dim)
        self.norm3 = nn.LayerNorm(hidden_dim)

        self.mlp = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 4),
            nn.GELU(),
            nn.Linear(hidden_dim * 4, hidden_dim),
        )

    def forward(
        self,
        x: torch.Tensor,
        past_key_values: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        encoder_hidden_states: Optional[torch.Tensor] = None,
        encoder_past_kv: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, Tuple[torch.Tensor, torch.Tensor], Optional[Tuple[torch.Tensor, torch.Tensor]]]:
        """
        Forward pass with self-attention and cross-attention.
        """
        batch_size, seq_len, _ = x.shape
        residual = x

        # 1. Self Attention
        q = self.self_attn_q(self.norm1(x)).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.self_attn_k(self.norm1(x)).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        v = self.self_attn_v(self.norm1(x)).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)

        if past_key_values is not None:
            prev_k, prev_v = past_key_values
            prev_k = prev_k.to(dtype=q.dtype)
            prev_v = prev_v.to(dtype=q.dtype)
            k = torch.cat([prev_k, k], dim=2)
            v = torch.cat([prev_v, v], dim=2)
        new_past_kv = (k, v)

        # Causal scale-dot product attention
        scale = 1.0 / math.sqrt(self.head_dim)
        attn_weights = torch.matmul(q, k.transpose(-2, -1)) * scale
        if attention_mask is not None:
            attn_weights = attn_weights + attention_mask.to(dtype=q.dtype)
        elif self.is_decoder and seq_len > 1:
            causal_mask = torch.triu(torch.full((seq_len, k.shape[2]), float("-inf"), device=x.device, dtype=q.dtype), diagonal=1)
            attn_weights = attn_weights + causal_mask

        attn_probs = F.softmax(attn_weights, dim=-1)
        self_attn_out = torch.matmul(attn_probs, v).transpose(1, 2).reshape(batch_size, seq_len, -1)
        x = residual + self.self_attn_out(self_attn_out)

        # 2. Cross Attention (if encoder state provided)
        new_cross_kv = None
        if encoder_hidden_states is not None or encoder_past_kv is not None:
            residual = x
            normed_x = self.norm2(x)
            cq = self.cross_attn_q(normed_x).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)

            if encoder_past_kv is not None:
                ck, cv = encoder_past_kv
                ck = ck.to(dtype=cq.dtype)
                cv = cv.to(dtype=cq.dtype)
            else:
                enc_len = encoder_hidden_states.shape[1]
                ck = self.cross_attn_k(encoder_hidden_states).view(batch_size, enc_len, self.num_heads, self.head_dim).transpose(1, 2)
                cv = self.cross_attn_v(encoder_hidden_states).view(batch_size, enc_len, self.num_heads, self.head_dim).transpose(1, 2)

            new_cross_kv = (ck, cv)
            cross_weights = torch.matmul(cq, ck.transpose(-2, -1)) * scale
            cross_probs = F.softmax(cross_weights, dim=-1)
            cross_attn_out = torch.matmul(cross_probs, cv).transpose(1, 2).reshape(batch_size, seq_len, -1)
            x = residual + self.cross_attn_out(cross_attn_out)

        # 3. MLP
        norm_layer = self.norm3 if (encoder_hidden_states is not None or encoder_past_kv is not None) else self.norm2
        x = x + self.mlp(norm_layer(x))
        return x, new_past_kv, new_cross_kv


class StreamingMultiCodebookTTSModel(nn.Module):
    """
    Phase 1: Neural-Codec-based TTS Model (Decoder-only autoregressive over K codebooks).
    Exposes HuggingFace-compatible config interface for seamless BlockManager initialization.
    """

    def __init__(
        self,
        num_layers: int = 4,
        num_heads: int = 4,
        hidden_dim: int = 256,
        num_codebooks: int = 4,
        vocab_size: int = 1024,
    ):
        super().__init__()
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.head_dim = hidden_dim // num_heads
        self.hidden_dim = hidden_dim
        self.num_codebooks = num_codebooks
        self.vocab_size = vocab_size

        # HuggingFace-like config wrapper for compatibility with BlockManager / Schedulers
        class Config:
            pass
        self.config = Config()
        self.config.num_hidden_layers = num_layers
        self.config.num_attention_heads = num_heads
        self.config.num_key_value_heads = num_heads
        self.config.hidden_size = hidden_dim
        self.config.vocab_size = vocab_size

        self.embeddings = MultiCodebookEmbedding(num_codebooks, vocab_size, hidden_dim)
        self.layers = nn.ModuleList([
            DecoderLayer(hidden_dim, num_heads, self.head_dim, is_decoder=True)
            for _ in range(num_layers)
        ])
        self.norm = nn.LayerNorm(hidden_dim)
        self.heads = MultiCodebookHeads(num_codebooks, hidden_dim, vocab_size)

    def forward(
        self,
        input_codebooks: torch.Tensor,
        past_key_values: Optional[Any] = None,
        use_cache: bool = True,
    ) -> Any:
        """
        Args:
            input_codebooks: [batch_size, seq_len, num_codebooks]
            past_key_values: DynamicCache or list of layer KV tuples
        Returns:
            Output object containing logits [batch_size, seq_len, num_codebooks, vocab_size] and past_key_values
        """
        x = self.embeddings(input_codebooks)

        out_cache = DynamicCache() if use_cache else None
        new_past_kv_list = []

        for i, layer in enumerate(self.layers):
            past_kv = None
            if past_key_values is not None:
                if isinstance(past_key_values, DynamicCache):
                    if i < len(past_key_values.layers):
                        past_kv = (past_key_values.layers[i].keys, past_key_values.layers[i].values)
                elif isinstance(past_key_values, (list, tuple)) and i < len(past_key_values):
                    past_kv = past_key_values[i]

            x, new_kv, _ = layer(x, past_key_values=past_kv)
            if use_cache:
                dyn_layer = DynamicLayer()
                dyn_layer.keys = new_kv[0]
                dyn_layer.values = new_kv[1]
                dyn_layer.is_initialized = True
                out_cache.layers.append(dyn_layer)
                new_past_kv_list.append(new_kv)

        x = self.norm(x)
        logits = self.heads(x)

        class ModelOutput:
            def __init__(self, logits, past_key_values):
                self.logits = logits
                self.past_key_values = past_key_values

        return ModelOutput(logits, out_cache if use_cache else None)


class WhisperClassSpeechEncoderDecoderModel(nn.Module):
    """
    Phase 2 & 3: Speech Encoder-Decoder Architecture (Whisper-class ASR model).
    Encodes audio features (mel frames) non-causally, then decodes text tokens causally with cross-attention.
    """

    def __init__(
        self,
        num_encoder_layers: int = 4,
        num_decoder_layers: int = 4,
        num_heads: int = 4,
        hidden_dim: int = 256,
        mel_dim: int = 80,
        vocab_size: int = 51200,
    ):
        super().__init__()
        self.num_encoder_layers = num_encoder_layers
        self.num_decoder_layers = num_decoder_layers
        self.num_heads = num_heads
        self.head_dim = hidden_dim // num_heads
        self.hidden_dim = hidden_dim
        self.vocab_size = vocab_size

        class Config:
            pass
        self.config = Config()
        self.config.num_hidden_layers = num_decoder_layers
        self.config.num_attention_heads = num_heads
        self.config.num_key_value_heads = num_heads
        self.config.hidden_size = hidden_dim
        self.config.vocab_size = vocab_size
        self.config.encoder_num_layers = num_encoder_layers

        # Audio encoder front-end
        self.conv1 = nn.Conv1d(mel_dim, hidden_dim, kernel_size=3, padding=1)
        self.conv2 = nn.Conv1d(hidden_dim, hidden_dim, kernel_size=3, stride=2, padding=1)
        self.encoder_layers = nn.ModuleList([
            DecoderLayer(hidden_dim, num_heads, self.head_dim, is_decoder=False)
            for _ in range(num_encoder_layers)
        ])
        self.encoder_norm = nn.LayerNorm(hidden_dim)

        # Decoder front-end
        self.decoder_embed = nn.Embedding(vocab_size, hidden_dim)
        self.decoder_layers = nn.ModuleList([
            DecoderLayer(hidden_dim, num_heads, self.head_dim, is_decoder=True)
            for _ in range(num_decoder_layers)
        ])
        self.decoder_norm = nn.LayerNorm(hidden_dim)
        self.lm_head = nn.Linear(hidden_dim, vocab_size, bias=False)

    def encode(self, mel_features: torch.Tensor) -> Tuple[torch.Tensor, List[Tuple[torch.Tensor, torch.Tensor]]]:
        """
        Non-causal audio feature encoder.
        Args:
            mel_features: [batch_size, mel_dim, mel_len]
        Returns:
            encoder_hidden_states: [batch_size, seq_len, hidden_dim]
            encoder_kv_list: List of (K, V) tuples per encoder layer [batch_size, num_heads, seq_len, head_dim]
        """
        x = F.gelu(self.conv1(mel_features))
        x = F.gelu(self.conv2(x)).transpose(1, 2) # [batch_size, seq_len, hidden_dim]

        encoder_kv_list = []
        for layer in self.encoder_layers:
            x, (k, v), _ = layer(x)
            encoder_kv_list.append((k, v))

        x = self.encoder_norm(x)
        return x, encoder_kv_list

    def decode_step(
        self,
        input_ids: torch.Tensor,
        past_key_values: Optional[Any] = None,
        encoder_past_kv_list: Optional[List[Tuple[torch.Tensor, torch.Tensor]]] = None,
        encoder_hidden_states: Optional[torch.Tensor] = None,
        use_cache: bool = True,
    ) -> Any:
        """
        Single token or prompt decode step with causal self-attention and cross-attention over encoder KV.
        """
        # nn.Embedding always returns float32; cast to match model weight dtype (e.g. float16)
        # to avoid cuBLAS GEMM dtype-mismatch failures on the first Linear projection.
        x = self.decoder_embed(input_ids).to(dtype=self.decoder_norm.weight.dtype)
        out_cache = DynamicCache() if use_cache else None

        for i, layer in enumerate(self.decoder_layers):
            past_kv = None
            if past_key_values is not None:
                if isinstance(past_key_values, DynamicCache):
                    if i < len(past_key_values.layers):
                        past_kv = (past_key_values.layers[i].keys, past_key_values.layers[i].values)
                elif isinstance(past_key_values, (list, tuple)) and i < len(past_key_values):
                    past_kv = past_key_values[i]

            enc_kv = encoder_past_kv_list[i] if encoder_past_kv_list is not None and i < len(encoder_past_kv_list) else None
            x, new_self_kv, _ = layer(
                x,
                past_key_values=past_kv,
                encoder_hidden_states=encoder_hidden_states if enc_kv is None else None,
                encoder_past_kv=enc_kv,
            )

            if use_cache:
                dyn_layer = DynamicLayer()
                dyn_layer.keys = new_self_kv[0]
                dyn_layer.values = new_self_kv[1]
                dyn_layer.is_initialized = True
                out_cache.layers.append(dyn_layer)

        x = self.decoder_norm(x)
        logits = self.lm_head(x)

        class ModelOutput:
            def __init__(self, logits, past_key_values):
                self.logits = logits
                self.past_key_values = past_key_values

        return ModelOutput(logits, out_cache if use_cache else None)
