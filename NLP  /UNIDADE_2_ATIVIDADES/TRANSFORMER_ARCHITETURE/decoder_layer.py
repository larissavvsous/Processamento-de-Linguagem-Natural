import torch.nn as nn
from unidade2.transformer_architeture.feedFowardSubLayer import FeedForwardSubLayer
from unidade2.transformer_architeture.attention_mecanism import MultiHeadAttention


class DecoderLayer(nn.Module):
    def __init__(self, d_model, n_heads, dim_feedforward=2048, dropout=0.1):
        super(DecoderLayer, self).__init__()
        self.self_attention = MultiHeadAttention(d_model, n_heads)
        self.cross_attention = MultiHeadAttention(d_model, n_heads)
        self.feed_forward = FeedForwardSubLayer(d_model, dim_feedforward, dropout)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, tgt, memory, tgt_mask=None, memory_mask=None):
        # Self attention
        tgt2 = self.self_attention(tgt, tgt, tgt, tgt_mask)
        tgt = self.norm1(tgt + self.dropout(tgt2))

        # Cross attention
        tgt2 = self.cross_attention(tgt, memory, memory, memory_mask)
        tgt = self.norm2(tgt + self.dropout(tgt2))

        # Feed Forward
        tgt = self.feed_forward(tgt)
        return tgt
