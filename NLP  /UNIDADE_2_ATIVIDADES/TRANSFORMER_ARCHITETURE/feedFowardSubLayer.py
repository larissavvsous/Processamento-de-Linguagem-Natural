import torch.nn as nn

class FeedForwardSubLayer(nn.Module):
    def __init__(self, d_model, dim_feedforward=2048, dropout=0.1):
        super(FeedForwardSubLayer, self).__init__()
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.relu = nn.ReLU()
        self.linear2 = nn.Linear(dim_feedforward, d_model)
        self.dropout = nn.Dropout(dropout)
        self.norm = nn.LayerNorm(d_model)

    def forward(self, x):
        residual = x
        x = self.linear1(x)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.linear2(x)
        x = self.dropout(x)
        x = self.norm(x + residual)  # Residual connection
        return x
