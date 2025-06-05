import torch.nn as nn
from torch import Tensor
from einops.layers.torch import Rearrange

# Patch Embedding
# This class is used to convert the input image into patches and then flatten them.  
class PatchEmbedding(nn.Module):
    def __init__(self, patch_size = 8, emb_size = 128):
        """
        Args:
            in_channels (int): Number of input channels (e.g., 3 for RGB images).
            patch_size (int): Size of the patches to be extracted from the input image.
            emb_size (int): Size of the embedding vector for each patch.
        """
        self.patch_size = patch_size
        super().__init__()
        self.projection = nn.Sequential(
            # Reshape the input tensor to create patches
            Rearrange('b c (n p) -> b n (p c)', p = patch_size), # Rearrange the input tensor to create patches
            nn.Linear(patch_size, emb_size) # Linear projection of the patch to the embedding size
        )

    def forward(self, x: Tensor) -> Tensor:
        x = x[:,:,:self.patch_size * (x.shape[-1] // self.patch_size)]
        x = self.projection(x)
        return x

# Multi-head Self-Attention
# This is a simplified version of the multi-head self-attention mechanism used in transformers.
class Attention(nn.Module):
    def __init__(self, dim, n_heads=8, dropout=0.):
        """
        Args: 
            dim (int): Dimension of the input features.
            heads (int): Number of attention heads.
            dropout (float): Dropout rate.
        """
         # Initialize the parent class
        super().__init__()
        self.heads = n_heads
        self.dim = dim
        self.scale = self.dim ** -0.5

        self.attention = nn.MultiheadAttention(embed_dim=dim, num_heads=n_heads, dropout=dropout, batch_first=True)

        self.query = nn.Linear(dim, dim, bias=False)
        self.key = nn.Linear(dim, dim, bias=False)  
        self.value = nn.Linear(dim, dim, bias=False)

    def forward(self, x):
        # x : (batch, seq_len, dim)
        q = self.query(x)
        k = self.key(x)
        v = self.value(x)
        attn_output, attn_output_weights = self.attention(q, k, v)
        
        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        
        return attn_output

# Normalization
class PreNorm(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.fn = fn
        self.ln = nn.LayerNorm(dim)

    def forward(self, x):
        return self.fn(self.ln(x))

# Feed Forward
class FeedForward(nn.Sequential):
    def __init__(self, dim, hidden_dim, dropout=0.):
        super().__init__(
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout)
        )

# Residual Connection
class Residual(nn.Module):
    def __init__(self, fn):
        super().__init__()
        self.fn = fn

    def forward(self, x, *args, **kwargs):
        return x + self.fn(x, *args, **kwargs)