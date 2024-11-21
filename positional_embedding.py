import torch.nn as nn
import torch

class PositionalEmbedding(nn.Module):
    def __init__(self, input_dim, output_dim, **kwargs):
        super(PositionalEmbedding, self).__init__()
        self.token_embedding = nn.Embedding(input_dim, output_dim)
        self.position_embedding = nn.Embedding(input_dim, output_dim)


    def forward(self, x):
        lenght = x.shape[-1]
        positions = torch.tensor(range(0, lenght)).cuda()
        embedded_tokens = self.token_embedding(x)
        embedded_positions = self.position_embedding(positions)
        return embedded_tokens + embedded_positions

