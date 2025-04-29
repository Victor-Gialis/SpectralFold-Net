import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import repeat
from utils import PatchEmbedding, Attention, PreNorm, FeedForward, Residual

class ViT(nn.Module):
    def __init__(self, ch=1, serie_len=2**14, patch_size=1024, emb_dim=32,
                n_layers=6, out_dim=2**14, dropout=0.1, heads=2):
        super(ViT, self).__init__()

        # Attributes
        self.channels = ch
        self.serie_len = serie_len
        self.patch_size = patch_size
        self.n_layers = n_layers

        # Patching
        self.patch_embedding = PatchEmbedding(in_channels=ch,
                                              patch_size=patch_size,
                                              emb_size=emb_dim)
        # Learnable params
        num_patches = (serie_len // patch_size) ** 2
        # Positional embeddings are not used in this implementation
        # self.pos_embedding = nn.Parameter(torch.randn(1, num_patches + 1, emb_dim))
        self.cls_token = nn.Parameter(torch.rand(1, 1, emb_dim))

        # Transformer Encoder
        self.layers = nn.ModuleList([])
        for _ in range(n_layers):
            transformer_block = nn.Sequential(
                Residual(PreNorm(emb_dim, Attention(emb_dim, n_heads = heads, dropout = dropout))),
                Residual(PreNorm(emb_dim, FeedForward(emb_dim, emb_dim, dropout = dropout))))
            self.layers.append(transformer_block)

        # Classification head
        self.head = nn.Sequential(nn.LayerNorm(emb_dim), nn.Linear(emb_dim, out_dim))

    def forward(self, serie):
        # Get patch embedding vectors
        x = self.patch_embedding(serie)
        b, n, _ = x.shape

        # Add cls token to inputs
        cls_tokens = repeat(self.cls_token, '1 1 d -> b 1 d', b = b)
        x = torch.cat([cls_tokens, x], dim=1)
        # x += self.pos_embedding[:, :(n + 1)]

        # Transformer layers
        for i in range(self.n_layers):
            x = self.layers[i](x)

        # Output based on classification token
        return self.head(x[:, 0, :])
