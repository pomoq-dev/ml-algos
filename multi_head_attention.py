import torch
import torch.nn as nn


class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, num_heads):
        super(MultiHeadAttention, self).__init__()

        # Define parameters
        self.num_heads = num_heads
        self.d_model = d_model
        self.depth = d_model // num_heads

        # Define linear layers for queries, keys, and values
        self.query_layer = nn.Linear(d_model, d_model)
        self.key_layer = nn.Linear(d_model, d_model)
        self.value_layer = nn.Linear(d_model, d_model)

        # Define output linear layer for concatenation
        self.output_layer = nn.Linear(d_model, d_model)

    def forward(self, query, key, value, mask=None):
        batch_size = query.size(0)

        # Apply linear layer to input for queries, keys, and values
        Q = self.query_layer(query)
        K = self.key_layer(key)
        V = self.value_layer(value)

        # Reshape queries, keys, and values for multi-head attention
        Q = Q.view(batch_size, -1, self.num_heads, self.depth)
        K = K.view(batch_size, -1, self.num_heads, self.depth)
        V = V.view(batch_size, -1, self.num_heads, self.depth)

        # Transpose dimensions for matrix multiplication
        Q = Q.transpose(1, 2)
        K = K.transpose(1, 2)
        V = V.transpose(1, 2)

        # Compute dot product of queries and keys for attention scores
        attention_scores = torch.matmul(Q, K.transpose(-1, -2))
        attention_scores /= torch.sqrt(torch.tensor(self.depth, dtype=torch.float32))

        # Apply optional mask to attention scores
        if mask is not None:
            mask = mask.unsqueeze(1)
            attention_scores = attention_scores.masked_fill(mask == 0, -1e9)

        # Apply softmax to attention scores to get attention weights
        attention_weights = torch.softmax(attention_scores, dim=-1)

        # Apply attention weights to values
        attention_output = torch.matmul(attention_weights, V)

        # Reshape and concatenate attention outputs
        attention_output = attention_output.transpose(1, 2)
        attention_output = attention_output.reshape(batch_size, -1, self.d_model)
        output = self.output_layer(attention_output)

        return output
