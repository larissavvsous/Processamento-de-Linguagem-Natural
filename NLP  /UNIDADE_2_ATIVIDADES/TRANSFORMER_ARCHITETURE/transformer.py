import torch.nn as nn
from unidade2.transformer_architeture.transformer_encoder import TransformerEncoder
from unidade2.transformer_architeture.transformer_decoder import TransformerDecoder

class Transformer(nn.Module):
    def __init__(self, d_model, n_heads, num_encoder_layers, num_decoder_layers, dim_feedforward=2048, dropout=0.1):
        super(Transformer, self).__init__()
        self.encoder = TransformerEncoder(d_model, n_heads, num_encoder_layers, dim_feedforward, dropout)
        self.decoder = TransformerDecoder(d_model, n_heads, num_decoder_layers, dim_feedforward, dropout)
        self.d_model = d_model
        self.n_heads = n_heads

    def forward(self, src, tgt, src_mask=None, tgt_mask=None, memory_mask=None):
        memory = self.encoder(src, src_mask)
        output = self.decoder(tgt, memory, tgt_mask, memory_mask)
        return output

# Exemplo de inicialização do modelo
if __name__ == "__main__":
    d_model = 512
    n_heads = 2
    num_encoder_layers = 6
    num_decoder_layers = 6
    transformer = Transformer(d_model, n_heads, num_encoder_layers, num_decoder_layers)
    print(transformer)
