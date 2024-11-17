import torch
import torch.nn as nn
from torch.optim import optimizer


class TransformerEncoder(nn.Module):
    def __init__(self, embed_dim, dense_dim, num_heads, **kwargs):
        super(TransformerEncoder, self).__init__()
        self.attension = nn.MultiheadAttention(embed_dim=embed_dim, num_heads=num_heads, batch_first=True)
        self.dense_proj = nn.Sequential(
            nn.Linear(embed_dim, dense_dim),
            nn.ReLU(),
            nn.Linear(dense_dim, embed_dim),
        )
        self.norm1 = nn.LayerNorm(embed_dim)
        self.norm2 = nn.LayerNorm(embed_dim)

    def forward(self, x, mask=None):
        attention_output, _ = self.attension(x, x, x, attn_mask=mask)
        proj_input = self.norm1(x + attention_output)
        proj_output = self.dense_proj(proj_input) 
        return self.norm2(proj_input + proj_output)