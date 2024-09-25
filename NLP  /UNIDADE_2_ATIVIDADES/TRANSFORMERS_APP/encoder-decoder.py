import torch
from unidade2.transformer_architeture.transformer_encoder import TransformerEncoder
from unidade2.transformer_architeture.transformer_decoder import TransformerDecoder

vocab_size = 10000
batch_size = 8
d_model = 256
num_heads = 4
num_layers = 4
d_ff = 256
sequence_length = 32
dropout = 0.1


input_sequence = torch.randint(0, vocab_size, (batch_size, sequence_length))
padding_mask = torch.randint(0, 2, (sequence_length, sequence_length))
causal_mask = torch.triu(torch.ones(sequence_length, sequence_length), diagonal=1)

encoder = TransformerEncoder(vocab_size, d_model, num_layers, num_heads, d_ff, sequence_length, dropout)
Decoder = TransformerDecoder(vocab_size, d_model, num_layers, num_heads, d_ff, sequence_length, dropout)

encoder_output = encoder(input_sequence, padding_mask)
decoder_output = decoder(input_sequence, causal_mask, encoder_output, padding_mask)

print("Batch's output shape: ", decoder_output.shape)
