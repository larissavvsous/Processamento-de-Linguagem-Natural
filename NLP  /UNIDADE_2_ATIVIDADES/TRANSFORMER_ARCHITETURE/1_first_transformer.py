import torch
import torch.nn as nn

class Transformer(nn.Module):
    def __init__(self, d_model=512, n_heads=2, num_encoder_layers=6, num_decoder_layers=6):
        super(Transformer, self).__init__()
        self.transformer = nn.Transformer(
            d_model=d_model,
            nhead=n_heads,
            num_encoder_layers=num_encoder_layers,
            num_decoder_layers=num_decoder_layers
        )

    def forward(self, src, tgt):
        output = self.transformer(src, tgt)
        return output

# Exemplo de inicialização
if __name__ == "__main__":
    model = Transformer()
    print(model)
