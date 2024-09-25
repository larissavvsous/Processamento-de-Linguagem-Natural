import torch.nn as nn
from unidade2.transformer_architeture.encoder_layer import EncoderLayer

class TransformerEncoder(nn.Module):
    def __init__(self, d_model, n_heads, num_layers, dim_feedforward=2048, dropout=0.1):
        super(TransformerEncoder, self).__init__()
        self.layers = nn.ModuleList([EncoderLayer(d_model, n_heads, dim_feedforward, dropout) for _ in range(num_layers)])
        self.norm = nn.LayerNorm(d_model)

    def forward(self, src, mask=None):
        for layer in self.layers:
            src = layer(src, mask)
        return self.norm(src)
