"""Lab 2 starter: scaled dot-product attention and multi-head attention."""

import math
import torch
from torch import nn


def attention(q, k, v, mask=None):
    # TODO(Lab 2): implement scaled dot-product attention.

    d_k = q.size(-1)

    # Compute scaled dot-product attention scores
    scores = torch.matmul(q, k.transpose(-2, -1))
    scores = scores / math.sqrt(d_k)

    # Apply mask if provided
    if mask is not None:
        if mask.dtype == torch.bool:
            scores = scores.masked_fill(~mask, float("-inf"))
        else:
            scores = scores + mask

    # Convert scores into attention weights
    weights = torch.softmax(scores, dim=-1)

    # Use the weights to combine the values
    output = torch.matmul(weights, v)

    return output, weights


class MultiHeadAttention(nn.Module):
    def __init__(self, *args, **kwargs):
        # TODO(Lab 2): define the projections/heads required by the notebook.
        super().__init__()

        d_model = kwargs.get(
            "d_model",
            args[0] if len(args) > 0 else None,
        )

        num_heads = kwargs.get(
            "num_heads",
            args[1] if len(args) > 1 else None,
        )

        if d_model is None or num_heads is None:
            raise TypeError("d_model and num_heads are required")

        if d_model % num_heads != 0:
            raise ValueError("d_model must be divisible by num_heads")

        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads

        # Q, K, V projections
        self.q_proj = nn.Linear(d_model, d_model)
        self.k_proj = nn.Linear(d_model, d_model)
        self.v_proj = nn.Linear(d_model, d_model)

        # Final output projection
        self.out_proj = nn.Linear(d_model, d_model)

    def forward(self, q, k, v, mask=None):
        batch_size = q.size(0)

        # Project Q, K, V
        q = self.q_proj(q)
        k = self.k_proj(k)
        v = self.v_proj(v)

        # Split into multiple heads
        q = q.view(
            batch_size,
            -1,
            self.num_heads,
            self.d_k
        ).transpose(1, 2)

        k = k.view(
            batch_size,
            -1,
            self.num_heads,
            self.d_k
        ).transpose(1, 2)

        v = v.view(
            batch_size,
            -1,
            self.num_heads,
            self.d_k
        ).transpose(1, 2)

        # Make mask compatible with the head dimension
        if mask is not None:
            if mask.dim() == 2:
                mask = mask.unsqueeze(0).unsqueeze(0)
            elif mask.dim() == 3:
                mask = mask.unsqueeze(1)

        # Apply attention to all heads
        x, _ = attention(q, k, v, mask)

        # Combine the heads again
        x = x.transpose(1, 2).contiguous()

        x = x.view(
            batch_size,
            -1,
            self.d_model
        )

        # Final linear projection
        output = self.out_proj(x)

        return output