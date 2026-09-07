"""
Phase 2: Encoder Attention Backend (Dense Bidirectional Tiled Attention).

Computes non-causal attention over paged encoder Key/Value memory blocks:
- Gathers non-contiguous physical blocks from key_pool and value_pool using encoder_block_table.
- Executes dense non-causal (bidirectional) scaled dot-product attention over all encoder sequence positions.
"""

import math
import torch
import torch.nn.functional as F
from typing import List, Tuple, Optional


class EncoderAttentionBackend:
    """
    Non-causal bidirectional paged attention backend.
    Gathers non-contiguous physical blocks for encoder KV and computes full attention.
    """

    @staticmethod
    def forward(
        query: torch.Tensor,
        encoder_block_table: List[int],
        key_pool_layer: torch.Tensor,
        value_pool_layer: torch.Tensor,
        encoder_seq_len: int,
        block_size: int = 16,
    ) -> torch.Tensor:
        """
        Compute non-causal bidirectional attention for a single layer.

        Args:
            query: [batch_size, num_heads, q_len, head_dim]
            encoder_block_table: Physical block indices storing the encoder KV sequence
            key_pool_layer: [num_blocks, block_size, num_heads, head_dim]
            value_pool_layer: [num_blocks, block_size, num_heads, head_dim]
            encoder_seq_len: Total active sequence length of the encoder
            block_size: Physical block size (tokens per block)

        Returns:
            output: [batch_size, num_heads, q_len, head_dim]
        """
        batch_size, num_heads, q_len, head_dim = query.shape

        if not encoder_block_table or encoder_seq_len == 0:
            return torch.zeros_like(query)

        # 1. Gather non-contiguous physical blocks from pool: [num_req_blocks, block_size, num_heads, head_dim]
        gathered_k = key_pool_layer[encoder_block_table]
        gathered_v = value_pool_layer[encoder_block_table]

        # 2. Flatten along token dimension and slice exact encoder_seq_len: [encoder_seq_len, num_heads, head_dim]
        flat_k = gathered_k.view(-1, num_heads, head_dim)[:encoder_seq_len]
        flat_v = gathered_v.view(-1, num_heads, head_dim)[:encoder_seq_len]

        # 3. Permute to layout: [1, num_heads, encoder_seq_len, head_dim]
        k_seq = flat_k.permute(1, 0, 2).unsqueeze(0)
        v_seq = flat_v.permute(1, 0, 2).unsqueeze(0)

        if batch_size > 1:
            k_seq = k_seq.expand(batch_size, -1, -1, -1)
            v_seq = v_seq.expand(batch_size, -1, -1, -1)

        # 4. Non-causal Scaled Dot-Product Attention
        scale = 1.0 / math.sqrt(head_dim)
        scores = torch.matmul(query, k_seq.transpose(-2, -1)) * scale # [batch_size, num_heads, q_len, encoder_seq_len]
        probs = F.softmax(scores, dim=-1)
        output = torch.matmul(probs, v_seq) # [batch_size, num_heads, q_len, head_dim]

        return output
