"""Lab 2 starter notebook-as-script.
Complete the marked sections, verify numerical equivalence, inspect parameter
accounting, causal masking, attention heads and pad-attention leakage.
"""

import math
import sys
from pathlib import Path

import torch
import torch.nn.functional as F

# Allow the script to find src when run using the README command.
PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.bayan.attention import attention, MultiHeadAttention


def main():
    torch.manual_seed(42)

    batch_size = 2
    seq_len = 5
    d_model = 8

    q = torch.randn(batch_size, seq_len, d_model)
    k = torch.randn(batch_size, seq_len, d_model)
    v = torch.randn(batch_size, seq_len, d_model)


    # --------------------------------------------------
    # 1. Verify numerical equivalence
    # --------------------------------------------------

    print("\n1. Numerical equivalence")

    output, weights = attention(q, k, v)

    # Calculate attention manually as a reference
    scores = torch.matmul(q, k.transpose(-2, -1))
    scores = scores / math.sqrt(d_model)

    expected_weights = F.softmax(scores, dim=-1)
    expected_output = torch.matmul(expected_weights, v)

    output_difference = (
        output - expected_output
    ).abs().max().item()

    weight_difference = (
        weights - expected_weights
    ).abs().max().item()

    print("Maximum output difference:", output_difference)
    print("Maximum weight difference:", weight_difference)

    assert torch.allclose(
        output,
        expected_output,
        atol=1e-6
    )

    assert torch.allclose(
        weights,
        expected_weights,
        atol=1e-6
    )

    print("PASS: attention matches the reference calculation")


    # --------------------------------------------------
    # 2. Inspect attention weight matrix
    # --------------------------------------------------

    print("\n2. Attention weight matrix")

    print(weights[0])

    row_sums = weights[0].sum(dim=-1)

    print("\nRow sums:")
    print(row_sums)

    assert torch.allclose(
        row_sums,
        torch.ones_like(row_sums),
        atol=1e-6
    )

    print("PASS: attention rows sum to 1")


    # --------------------------------------------------
    # 3. Exercise Multi-Head Attention
    # --------------------------------------------------

    print("\n3. Multi-Head Attention")

    num_heads = 2

    mha = MultiHeadAttention(
        d_model=d_model,
        num_heads=num_heads
    )

    mha_output = mha(q, k, v)

    print("Number of heads:", num_heads)
    print("Head dimension:", d_model // num_heads)
    print("Input shape:", q.shape)
    print("Output shape:", mha_output.shape)

    assert mha_output.shape == q.shape

    print("PASS: Multi-Head Attention output shape is correct")


    # --------------------------------------------------
    # 4. Parameter accounting
    # --------------------------------------------------

    print("\n4. Parameter accounting")

    parameter_count = sum(
        parameter.numel()
        for parameter in mha.parameters()
    )

    # Four linear layers:
    # Q, K, V, and the final output projection
    expected_parameters = 4 * (
        d_model * d_model + d_model
    )

    print("Actual parameters:", parameter_count)
    print("Expected parameters:", expected_parameters)

    assert parameter_count == expected_parameters

    print("PASS: parameter count is correct")


    # --------------------------------------------------
    # 5. Verify padding mask behaviour
    # --------------------------------------------------

    print("\n5. Padding mask")

    # True means attention is allowed.
    pad_mask = torch.ones(
        batch_size,
        seq_len,
        seq_len,
        dtype=torch.bool
    )

    # Pretend the final two tokens are padding.
    pad_mask[:, :, -2:] = False

    _, masked_weights = attention(
        q,
        k,
        v,
        mask=pad_mask
    )

    print(masked_weights[0])

    pad_attention_mass = (
        masked_weights[..., -2:].sum().item()
    )

    print(
        "Attention mass on padded positions:",
        pad_attention_mass
    )

    assert pad_attention_mass < 1e-6

    print("PASS: padded positions receive zero attention")


    # --------------------------------------------------
    # 6. Verify causal masking
    # --------------------------------------------------

    print("\n6. Causal mask")

    # Lower-triangular mask prevents attention
    # to future tokens.
    causal_mask = torch.tril(
        torch.ones(
            seq_len,
            seq_len,
            dtype=torch.bool
        )
    )

    _, causal_weights = attention(
        q,
        k,
        v,
        mask=causal_mask
    )

    print(causal_weights[0])

    future_attention = torch.triu(
        causal_weights[0],
        diagonal=1
    ).sum().item()

    print(
        "Attention mass on future positions:",
        future_attention
    )

    assert future_attention < 1e-6

    print("PASS: tokens cannot attend to future tokens")


if __name__ == "__main__":
    main()