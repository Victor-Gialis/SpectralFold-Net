import torch
import torch.nn as nn
from utils.transformer_blocks import PatchEmbedding, Attention, PreNorm, FeedForward, Residual

class ViTAutoencoder(nn.Module):
    def __init__(self, encoder_dim=2**10, decoder_dim=2**8, serie_len=8192, patch_size=128, dropout=0.1, n_layers=6, heads=2):
        super().__init__()
        self.num_patch = serie_len // patch_size

        # Patch embedding
        self.patch_embedding = PatchEmbedding(patch_size=patch_size, emb_size=encoder_dim)

        # Positional embedding and cls token
        self.position_embedding = nn.Parameter(torch.zeros(1, (serie_len // patch_size) + 1, encoder_dim), requires_grad=False)  # +1 for cls token
        self.cls_token = nn.Parameter(torch.zeros(1, 1, encoder_dim))

        # Encoder
        self.encoder_layers = nn.ModuleList([
            nn.Sequential(
                Residual(PreNorm(encoder_dim, Attention(encoder_dim, n_heads=heads, dropout=dropout))),
                Residual(PreNorm(encoder_dim, FeedForward(encoder_dim, encoder_dim, dropout=dropout)))
            )
            for _ in range(n_layers)
        ])
        self.encoder_output_head = nn.Sequential(
            nn.LayerNorm(encoder_dim),
            nn.Linear(encoder_dim, encoder_dim),
            nn.Dropout(dropout)
        )

        # Decoder
        self.mask_token = nn.Parameter(torch.zeros(1, 1, encoder_dim))  # Mask token for padding

        self.decoder_projection = nn.Sequential(
            nn.LayerNorm(encoder_dim),
            nn.Linear(encoder_dim, decoder_dim),
            nn.Dropout(dropout)
        )
        self.decoder_layers = nn.ModuleList([
            nn.Sequential(
                Residual(PreNorm(decoder_dim, Attention(decoder_dim, n_heads=heads, dropout=dropout))),
                Residual(PreNorm(decoder_dim, FeedForward(decoder_dim, decoder_dim, dropout=dropout)))
            )
            for _ in range(n_layers)
        ])
        self.decoder_output_head = nn.Sequential(
            nn.LayerNorm(decoder_dim),
            nn.Linear(decoder_dim, patch_size),
            nn.GELU(),
            nn.Linear(patch_size, patch_size),
            nn.Flatten(start_dim=1),
        )

        self._initialize_weights()
    
    def _initialize_weights(self):
        """Initialize weights."""
        # Initialize encoder and decoder layers
        # using Xavier uniform initialization for linear layers
        for layer in self.encoder_layers:
            for sublayer in layer:
                if isinstance(sublayer, nn.Linear):
                    nn.init.xavier_uniform_(sublayer.weight)
                    if sublayer.bias is not None:
                        nn.init.zeros_(sublayer.bias)

        for layer in self.decoder_layers:
            for sublayer in layer:
                if isinstance(sublayer, nn.Linear):
                    nn.init.xavier_uniform_(sublayer.weight)
                    if sublayer.bias is not None:
                        nn.init.zeros_(sublayer.bias)

        nn.init.normal_(self.cls_token, std=0.02)
        nn.init.normal_(self.mask_token, std=0.02)

        # Initialize positional embeddings sincos 
        position = torch.arange(0, self.position_embedding.shape[1], dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, self.position_embedding.shape[2], 2).float() * -(torch.log(torch.tensor(10000.0)) / self.position_embedding.shape[2]))
        self.position_embedding[:, :, 0::2] = torch.sin(position * div_term)
        self.position_embedding[:, :, 1::2] = torch.cos(position * div_term)
        self.position_embedding.requires_grad = False  # Make sure positional embeddings are not trainable

        # Initialize patch embedding weights
        nn.init.xavier_uniform_(self.patch_embedding.projection[-1].weight)
        if self.patch_embedding.projection[-1].bias is not None:
            nn.init.zeros_(self.patch_embedding.projection[-1].bias)

    def forward_encoder(self, x):
        """Encodes the input using the encoder."""
        b, n, _ = x.shape

        # Add cls token
        cls_tokens = self.cls_token.expand(b, -1, -1)  # Étendre le cls_token pour correspondre à la taille du batch
        x = torch.cat([cls_tokens, x], dim=1)  # Concaténer le cls_token avec les embeddings des patches

        # Add positional embedding
        x = x + self.position_embedding[:, :n + 1, :]

        # Encoder layers
        for layer in self.encoder_layers:
            x = layer(x)

        # Encoder output
        encoded_tokens = self.encoder_output_head(x)
        return encoded_tokens

    def forward_decoder(self, encoded_tokens, n, b):
        """Decodes the encoded tokens using the decoder."""

        # Padding patch tokens using mask token
        if n < self.num_patch:
            bottom_padding = self.mask_token.expand(b, self.num_patch - n, -1)  # Create padding tokens
            bottom_padding = bottom_padding.to(encoded_tokens.device)  # Move to the same device as x
            x = torch.  cat([encoded_tokens, bottom_padding], dim=1)  # bottom padding

        # Add positional embedding
        x = x + self.position_embedding

        # Decoder projection
        x = self.decoder_projection(x)

        # Decoder layers
        for layer in self.decoder_layers:
            x = layer(x)

        # Remove cls token
        x = x[:, 1:]  # [B, N_patches, emb_dim]

        # Decoder output
        decoded_tokens = self.decoder_output_head(x)
        return decoded_tokens

    def forward(self, x):
        """Forward method for the autoencoder."""
        # Patch embedding
        patch_emb = self.patch_embedding(x)
        b, n, _ = patch_emb.shape

        # Forward through encoder
        encoded_tokens = self.forward_encoder(patch_emb)

        # Forward through decoder
        decoded_tokens = self.forward_decoder(encoded_tokens, n, b)

        return patch_emb, encoded_tokens, decoded_tokens