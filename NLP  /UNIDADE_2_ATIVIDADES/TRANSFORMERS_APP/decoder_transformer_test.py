import torch
from unidade2.transformer_architeture.transformer_decoder import TransformerDecoder
vocab_size = 10000
batch_size = 8
d_model = 256
num_heads = 16
num_layers = 8
d_ff = 512
sequence_length = 128
dropout = 0.1

input_sequence = torch.randint(0, vocab_size, (batch_size, sequence_length))

encoder_output = torch.rand(batch_size, sequence_length, d_model)

self_attention_mask = (1 - torch.triu(torch.ones(sequence_length, sequence_length), diagonal=1)).bool()

cross_attention_mask = None

decoder = TransformerDecoder(vocab_size, d_model, num_layers, num_heads, d_ff, sequence_length, dropout)

output = decoder(input_sequence, self_attention_mask, encoder_output, cross_attention_mask)

print(output.shape)
