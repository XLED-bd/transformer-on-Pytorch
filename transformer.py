import torch.nn as nn
from transformer_encoder import TransformerEncoder
import torch

class Transformer(nn.Module):
    def __init__(self, embed_dim, dense_dim, num_heads, vocab_size, **kwargs):
        super(Transformer, self).__init__()

        self.embedding = nn.Embedding(vocab_size, embed_dim)


        self.encoder = TransformerEncoder(embed_dim, dense_dim, num_heads)
        # self.decoder = 

        self.global_max_pool = nn.AdaptiveMaxPool1d(1)
        self.out = nn.Linear(600, 1)
        self.sigmoid = nn.Sigmoid()


    def forward(self, text, mask=None):
        embedded = self.embedding(text)
        encoder_output = self.encoder(embedded, mask)
        # decoder_output = self.decoder(encoder_output, mask)
        output = self.global_max_pool(encoder_output)
        output = output.squeeze(-1)  # (batch_size, embed_dim) -> (batch_size, 1)
        output = self.out(output)
        output = self.sigmoid(output)
        return output
