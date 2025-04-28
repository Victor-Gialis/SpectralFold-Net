import torch
import numpy as np

from einops import rearrange, repeat
from einops.layers.torch import Rearrange

from timm.models.layers import trunc_normal_
from timm.models.vision_transformer import Block

def random_indexes(size: int):
    forward_indexes = np.arange(size)
    np.random.shuffle(forward_indexes)
    backward_indexes = np.argsort(forward_indexes)
    return forward_indexes, backward_indexes

def take_indexes(sequences, indexes):
    return torch.gather(sequences, 0, repeat(indexes, 't b -> t b c', c=sequences.shape[-1]))

class PatchShuffle(torch.nn.Module):
    def __init__(self, ratio):
        super().__init__()
        self.ratio = ratio

    def forward(self, patches: torch.Tensor):
        T, B, C = patches.shape
        remain_T = int(T * (1 - self.ratio))

        indexes = [random_indexes(T) for _ in range(B)]
        forward_indexes = torch.as_tensor(np.stack([i[0] for i in indexes], axis=-1), dtype=torch.long).to(patches.device)
        backward_indexes = torch.as_tensor(np.stack([i[1] for i in indexes], axis=-1), dtype=torch.long).to(patches.device)

        patches = take_indexes(patches, forward_indexes)
        patches = patches[:remain_T]

        return patches, forward_indexes, backward_indexes

class MAE1D_Encoder(torch.nn.Module):
    def __init__(self, 
                 sequence_length=12048,
                 patch_size=16,
                 emb_dim=192,
                 num_layer=12,
                 num_head=3,
                 mask_ratio=0.75):
        super().__init__()

        self.cls_token = torch.nn.Parameter(torch.zeros(1, 1, emb_dim))
        self.num_patches = sequence_length // patch_size
        self.pos_embedding = torch.nn.Parameter(torch.zeros(self.num_patches, 1, emb_dim))
        self.shuffle = PatchShuffle(mask_ratio)

        self.patchify = torch.nn.Conv1d(1, emb_dim, patch_size, patch_size)  # 1D

        self.transformer = torch.nn.Sequential(*[Block(emb_dim, num_head) for _ in range(num_layer)])
        self.layer_norm = torch.nn.LayerNorm(emb_dim)

        self.init_weight()

    def init_weight(self):
        trunc_normal_(self.cls_token, std=.02)
        trunc_normal_(self.pos_embedding, std=.02)

    def forward(self, x):
        # x : (batch, 1, seq_len)
        patches = self.patchify(x)  # (batch, emb_dim, num_patches)
        patches = rearrange(patches, 'b c t -> t b c')  # (tokens, batch, embed_dim)
        patches = patches + self.pos_embedding

        patches, forward_indexes, backward_indexes = self.shuffle(patches)

        patches = torch.cat([self.cls_token.expand(-1, patches.shape[1], -1), patches], dim=0)
        patches = rearrange(patches, 't b c -> b t c')
        features = self.layer_norm(self.transformer(patches))
        features = rearrange(features, 'b t c -> t b c')

        return features, backward_indexes

class MAE1D_Decoder(torch.nn.Module):
    def __init__(self, 
                 sequence_length=12048,
                 patch_size=16,
                 emb_dim=192,
                 num_layer=4,
                 num_head=3):
        super().__init__()

        self.num_patches = sequence_length // patch_size
        self.mask_token = torch.nn.Parameter(torch.zeros(1, 1, emb_dim))
        self.pos_embedding = torch.nn.Parameter(torch.zeros(self.num_patches + 1, 1, emb_dim))

        self.transformer = torch.nn.Sequential(*[Block(emb_dim, num_head) for _ in range(num_layer)])
        self.head = torch.nn.Linear(emb_dim, patch_size)

        self.init_weight()

    def init_weight(self):
        trunc_normal_(self.mask_token, std=.02)
        trunc_normal_(self.pos_embedding, std=.02)

    def forward(self, features, backward_indexes):
        T = features.shape[0]

        backward_indexes = torch.cat([
            torch.zeros(1, backward_indexes.shape[1], dtype=torch.long).to(backward_indexes.device),
            backward_indexes + 1
        ], dim=0)

        features = torch.cat([
            features,
            self.mask_token.expand(backward_indexes.shape[0] - features.shape[0], features.shape[1], -1)
        ], dim=0)

        features = take_indexes(features, backward_indexes)
        features = features + self.pos_embedding

        features = rearrange(features, 't b c -> b t c')
        features = self.transformer(features)
        features = rearrange(features, 'b t c -> t b c')

        features = features[1:]  # remove global token
        patches = self.head(features)

        # Reconstruction
        recon = rearrange(patches, 't b p -> b 1 (t p)')

        # Mask
        mask = torch.zeros_like(patches)
        mask[T-1:] = 1
        mask = take_indexes(mask, backward_indexes[1:] - 1)
        mask = rearrange(mask, 't b p -> b 1 (t p)')

        return recon, mask

class MAE1D(torch.nn.Module):
    def __init__(self, 
                 sequence_length=12048,
                 patch_size=16,
                 emb_dim=192,
                 encoder_layer=12,
                 encoder_head=3,
                 decoder_layer=4,
                 decoder_head=3,
                 mask_ratio=0.75):
        super().__init__()

        self.encoder = MAE1D_Encoder(sequence_length, patch_size, emb_dim, encoder_layer, encoder_head, mask_ratio)
        self.decoder = MAE1D_Decoder(sequence_length, patch_size, emb_dim, decoder_layer, decoder_head)

    def forward(self, x):
        features, backward_indexes = self.encoder(x)
        recon, mask = self.decoder(features, backward_indexes)
        return recon, mask
