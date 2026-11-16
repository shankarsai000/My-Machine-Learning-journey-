#Implementation 1 — Build self-attention using NumPy
print("Implementation 1 — Build self-attention using NumPy")
import numpy as np

np.random.seed(42)

# Four tokens, each represented by four numbers
X = np.array([
    [0.2, 0.5, 0.1, 0.7],
    [0.8, 0.3, 0.4, 0.2],
    [0.6, 0.9, 0.2, 0.5],
    [0.7, 0.4, 0.8, 0.3]
])

print("Input shape:", X.shape)
print(X)

d_model = 4
d_k = 4
d_v = 4

W_Q = np.random.randn(d_model, d_k)
W_K = np.random.randn(d_model, d_k)
W_V = np.random.randn(d_model, d_v)

Q = X @ W_Q
K = X @ W_K
V = X @ W_V

print("Q shape:", Q.shape)
print("K shape:", K.shape)
print("V shape:", V.shape)

scores = Q @ K.T

print("Attention scores:")
print(np.round(scores, 3))

scaled_scores = scores / np.sqrt(d_k)

print("Scaled scores:")
print(np.round(scaled_scores, 3))


def softmax(x, axis=-1):
    """Compute numerically stable softmax."""
    shifted_x = x - np.max(x, axis=axis, keepdims=True)
    exp_x = np.exp(shifted_x)

    return exp_x / np.sum(
        exp_x,
        axis=axis,
        keepdims=True
    )


attention_weights = softmax(scaled_scores)

print("Attention weights:")
print(np.round(attention_weights, 3))

print(
    "Row sums:",
    np.round(attention_weights.sum(axis=-1), 3)
)

attention_output = attention_weights @ V

print("Final attention output:")
print(np.round(attention_output, 3))

print("Output shape:", attention_output.shape)

#Implementation 2 — Self-attention using PyTorch
print("Implementation 2 — Self-attention using PyTorch")

import torch
import torch.nn as nn
import torch.nn.functional as F


class SelfAttention(nn.Module):
    def __init__(self, d_model):
        super().__init__()

        self.query = nn.Linear(d_model, d_model)
        self.key = nn.Linear(d_model, d_model)
        self.value = nn.Linear(d_model, d_model)

        self.scale = d_model ** 0.5

    def forward(self, x):
        # Generate Query, Key and Value
        Q = self.query(x)
        K = self.key(x)
        V = self.value(x)

        # Calculate scaled attention scores
        scores = torch.matmul(
            Q,
            K.transpose(-2, -1)
        ) / self.scale

        # Convert scores into attention weights
        attention_weights = F.softmax(
            scores,
            dim=-1
        )

        # Weighted combination of Value vectors
        output = torch.matmul(
            attention_weights,
            V
        )

        return output, attention_weights


# Example input: batch size 1, 4 tokens, 8 features
x = torch.randn(1, 4, 8)

attention = SelfAttention(d_model=8)

output, weights = attention(x)

print("Input shape:", x.shape)
print("Output shape:", output.shape)
print("Attention weights shape:", weights.shape)

#Implementation 3 — Multi-head attention using PyTorch
print("Implementation 3 — Multi-head attention using PyTorch")

class MultiHeadSelfAttention(nn.Module):
    def __init__(self, d_model, num_heads):
        super().__init__()

        if d_model % num_heads != 0:
            raise ValueError(
                "d_model must be divisible by num_heads."
            )

        self.attention = nn.MultiheadAttention(
            embed_dim=d_model,
            num_heads=num_heads,
            batch_first=True
        )

    def forward(self, x):
        output, attention_weights = self.attention(
            x, x, x,
            need_weights=True,
            average_attn_weights=False
        )

        return output, attention_weights


x = torch.randn(2, 5, 16)

model = MultiHeadSelfAttention(
    d_model=16,
    num_heads=4
)

output, weights = model(x)

print("Output shape:", output.shape)
print("Attention weights:", weights.shape)