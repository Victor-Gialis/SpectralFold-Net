import torch
import torch.nn as nn
from utils.transformer_blocks import PatchEmbedding, Attention, PreNorm, FeedForward, Residual

class ViTEncoder(nn.Module):
    def __init__(self, emb_dim=2**11, n_layers=2, dropout=0.1, heads=4):
        super().__init__()

        # parameters
        self.emb_dim = emb_dim
        self.n_layers = n_layers
        self.dropout = dropout
        self.heads = heads

        # Transformer Encoder
        self.layers = nn.ModuleList([])
        for _ in range(n_layers):
            transformer_block = nn.Sequential(
                Residual(PreNorm(emb_dim, Attention(emb_dim, n_heads = heads, dropout = dropout))),
                Residual(PreNorm(emb_dim, FeedForward(emb_dim, emb_dim, dropout = dropout))))
            self.layers.append(transformer_block)

        self.output_head = nn.Sequential(
            nn.LayerNorm(emb_dim),
            nn.Linear(emb_dim, emb_dim),
            nn.Dropout(dropout))

    def forward(self, x):
        # get patch embedding vectors
        b, n, d = x.shape

        # Transformer layers
        for i in range(self.n_layers):
            x = self.layers[i](x)

        x = self.output_head(x)  # [B, N_patches, emb_dim]
        return x

class ViTDecoder(nn.Module):
    def __init__(self, encoder_dim = 256, decoder_dim = 512, patch_size = 128, serie_len = 8192, n_layers=2, dropout=0.1, heads=4, num_patch=int):
        super().__init__()

        # parameters
        self.encoder_dim = encoder_dim
        self.decoder_dim = decoder_dim
        self.patch_size = patch_size
        self.serie_len = serie_len
        self.n_layers = n_layers
        self.num_patch = num_patch 

        self.projection = nn.Sequential(
            nn.LayerNorm(encoder_dim),
            nn.Linear(encoder_dim, decoder_dim),
            nn.Dropout(dropout))
        
        # Transformer Encoder
        self.layers = nn.ModuleList([])
        for _ in range(n_layers):
            transformer_block = nn.Sequential(
                Residual(PreNorm(decoder_dim, Attention(decoder_dim, n_heads = heads, dropout = dropout))),
                Residual(PreNorm(decoder_dim, FeedForward(decoder_dim, decoder_dim, dropout = dropout))))
            self.layers.append(transformer_block)
        
        self.output_head = nn.Sequential(
            nn.LayerNorm(decoder_dim),
            nn.Linear(decoder_dim, patch_size),
            nn.GELU(),
            nn.Linear(patch_size, patch_size),
            nn.Flatten(start_dim=1),
        )
    
    def forward(self, x):
        b, n, d = x.shape

        # Padding patch tokens
        if n < self.num_patch :
            bottom_padding = torch.randn(b,self.num_patch - n,d) # (b, n_patches - n, emb_dim)
            bottom_padding = bottom_padding.to(x.device) # Move to the same device as x
            x = torch.cat([x,bottom_padding], dim=1) # bottom padding       

        # get patch embedding vectors
        x = self.projection(x)  # (b, n, d)

        # Transformer layers
        for i in range(self.n_layers):
            x = self.layers[i](x)

        # remove cls token
        x = x[:,1:]  # [B, N_patches, emb_dim]

        return self.output_head(x)
    
class ViTAutoencoder(nn.Module):
    def __init__(self, encoder_dim = 2**10, decoder_dim = 2**8, serie_len = 8192, patch_size = 128, dropout = 0.1, n_layers = 6, heads = 2):
        super().__init__()
        self.num_patch = serie_len // patch_size
        # Encoder-Decoder
        self.patch_embedding = PatchEmbedding(patch_size=patch_size, emb_size=encoder_dim)
        self.position_embedding = nn.Parameter(torch.randn(1, (serie_len // patch_size)+1, encoder_dim))
        self.cls_token = nn.Parameter(torch.rand(1, 1, encoder_dim))
        self.encoder = ViTEncoder(encoder_dim, n_layers, dropout, heads)
        self.decoder = ViTDecoder(encoder_dim, decoder_dim, patch_size, serie_len, n_layers, dropout, heads, num_patch=self.num_patch)

    def forward(self, x):
        # Get patch embedding vectors
        patch_emb = self.patch_embedding(x)
        b, n, _ = patch_emb.shape

        # Add cls token
        cls_tokens = self.cls_token.expand(b, -1, -1)  # Étendre le cls_token pour correspondre à la taille du batch
        x1 = torch.cat([cls_tokens, patch_emb], dim=1)  # Concaténer le cls_token avec les embeddings des patches

        # Add positional embedding
        x1 = x1 + self.position_embedding[:, :n + 1, :]

        # Encoder
        encoded_tokens = self.encoder(x1)

        # Add positional embedding
        x2 = encoded_tokens + self.position_embedding[:, :n + 1, :]

        # Decoder
        decoded_tokens = self.decoder(x2)

        return patch_emb, encoded_tokens, decoded_tokens

