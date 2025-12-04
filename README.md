This homework implements two major components used in Transformer-based architectures:

Scaled Dot-Product Attention using NumPy

Transformer Encoder Block in PyTorch, including:

Multi-head self-attention

Residual connections

Layer normalization

Feed-forward network

Output shape verification

The purpose of this assignment is to understand how attention is computed and how encoder blocks process token sequences.

1️⃣ Scaled Dot-Product Attention (NumPy)
Goal:

Implement the attention formula:

Attention
(
𝑄
,
𝐾
,
𝑉
)
=
softmax
(
𝑄
𝐾
⊤
𝑑
𝑘
)
𝑉
Attention(Q,K,V)=softmax(
d
k
	​

	​

QK
⊤
	​

)V
Key Features:

Computes similarity scores between queries and keys

Applies softmax to convert scores into attention weights

Supports optional masking

Computes context vectors using weighted sum of values

Code Implementation
import numpy as np

def softmax(x, axis=-1):
    x_shifted = x - np.max(x, axis=axis, keepdims=True)
    exp_x = np.exp(x_shifted)
    return exp_x / np.sum(exp_x, axis=axis, keepdims=True)

def scaled_dot_product_attention(Q, K, V, mask=None):
    d_k = K.shape[-1]

    scores = np.matmul(Q, np.transpose(K, (0, 2, 1))) / np.sqrt(d_k)

    if mask is not None:
        scores = scores + (mask * -1e9)

    attention_weights = softmax(scores, axis=-1)
    context = np.matmul(attention_weights, V)

    return attention_weights, context

2️⃣ Transformer Encoder Block (PyTorch)
Goal:

Build a simplified version of a Transformer encoder block using:

d_model = 128

num_heads (h) = 8

feed-forward dimension = 512

Includes:

✔ Multi-head self-attention
✔ Add & Norm (residual + layer normalization)
✔ Feed-forward network
✔ Dropout regularization

Code Implementation
import torch
import torch.nn as nn

class TransformerEncoderBlock(nn.Module):
    def __init__(self, d_model=128, num_heads=8, d_ff=512, dropout=0.1):
        super().__init__()

        # Multi-head self-attention
        self.self_attn = nn.MultiheadAttention(
            embed_dim=d_model,
            num_heads=num_heads,
            batch_first=True
        )

        # Feed-forward network
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.ReLU(),
            nn.Linear(d_ff, d_model)
        )

        # Layer normalization
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)

        # Dropout layers
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

    def forward(self, x, attn_mask=None, key_padding_mask=None):

        # Self-attention
        attn_output, _ = self.self_attn(x, x, x)
        x = self.norm1(x + self.dropout1(attn_output))

        # Feed-forward pass
        ffn_output = self.ffn(x)
        x = self.norm2(x + self.dropout2(ffn_output))

        return x

3️⃣ Output Shape Verification
Task:

Verify the output shape for:

Batch size: 32

Sequence length: 10 tokens

Embedding dimension: 128

Test Code:
batch_size = 32
seq_len = 10
d_model = 128

encoder_block = TransformerEncoderBlock(d_model=d_model, num_heads=8, d_ff=512)

x = torch.randn(batch_size, seq_len, d_model)
output = encoder_block(x)

print(output.shape)

Expected Output:
torch.Size([32, 10, 128])


This confirms that the encoder block preserves the sequence and embedding dimensions.

📂 Project Structure
Homework5/
│── README.md
│── attention_numpy.py
│── encoder_block.py
│── test_encoder.py

📝 How to Run the Files
Run attention implementation:
python attention_numpy.py

Run encoder block test:
python test_encoder.py

Requirements:

Python 3.8+

PyTorch

NumPy

🎯 Learning Outcomes

By completing this assignment, you demonstrate:

✔ Understanding of scaled dot-product attention
✔ Ability to implement fundamental Transformer components
✔ Knowledge of residual connections and layer normalization
✔ Ability to verify tensor shapes and batching
✔ Practical experience using NumPy and PyTorch
