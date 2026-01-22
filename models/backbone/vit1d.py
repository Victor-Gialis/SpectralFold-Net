import torch
import torch.nn as nn
from types import SimpleNamespace
from utils.transformer_blocks import Tokeniser, Attention, PreNorm, FeedForward, Residual, PositionalEncoding

class ViT1DEncoder(nn.Module):
    def __init__(self, patch_size=16, hidden_dim=512, n_layers=3, heads=8, dropout=0.2565):
        super().__init__()
        # Encoder hyperparameters
        self.patch_size = patch_size
        self.hidden_dim = hidden_dim
        self.n_layers = n_layers
        self.heads = heads
        self.dropout = dropout

        # Pretrain dataset Stats
        self.stats = None

        # Tokeniser
        self.tokeniser = Tokeniser(patch_size=patch_size, embed_dim=hidden_dim)

        # Positional embedding
        self.positional_embedding = PositionalEncoding(hidden_dim= self.hidden_dim)

        # Class token
        self.cls_token = nn.Parameter(torch.zeros(1, 1, hidden_dim))

        # Encoder parameters
        self.encoder_layers = nn.ModuleList([
            nn.Sequential(
                Residual(PreNorm(hidden_dim, Attention(hidden_dim, n_heads=heads, dropout=dropout))),
                Residual(PreNorm(hidden_dim, FeedForward(hidden_dim, hidden_dim, dropout=dropout)))
            )
            for _ in range(n_layers)
        ])
        self.encoder_output_head = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout)
        )

        self._initialize_weights()

    def _initialize_weights(self):
        # Initialize cls token
        nn.init.normal_(self.cls_token, std=0.02)

        # Using Xavier uniform initialization for linear layers
        for layer in self.encoder_layers:
            for sublayer in layer:
                if isinstance(sublayer, nn.Linear):
                    nn.init.xavier_uniform_(sublayer.weight)
                    if sublayer.bias is not None:
                        nn.init.zeros_(sublayer.bias)
    
    def forward_tokeniser(self, x):
        return self.tokeniser(x)
    
    def forward_encoder(self, x):
        """Encodes the input using the encoder."""
        # Extract batch size and number of tokens
        b, _, _ = x.shape
        # Add cls token
        cls_tokens = self.cls_token.expand(b, -1, -1)  # Étendre le cls_token pour correspondre à la taille du batch
        x = torch.cat([cls_tokens, x], dim=1)  # Concaténer le cls_token avec les embeddings des patches
        # Add positional embedding
        x = self.positional_embedding(x)
        # Encoder layers
        for layer in self.encoder_layers:
            x = layer(x)
        # Encoder output
        encoded_tokens = self.encoder_output_head(x)
        return encoded_tokens
    
    def forward(self, x):
        tokens = self.forward_tokeniser(x)
        embeded_tokens = self.forward_encoder(tokens)
        return embeded_tokens
    
    def _loads_stats(self, stats):
        self.stats = stats

    def _get_arguments(self):
        """
        Return the arguments of the model.
        Returns:
            args (SimpleNamespace): Arguments of the model
        """
        return SimpleNamespace(
            patch_size = self.patch_size,
            hidden_dim = self.hidden_dim,
            n_layers = self.n_layers,
            heads = self.heads,
            dropout = self.dropout,
            )
    
class ViT1DDecoder(nn.Module):
    def __init__(self, patch_size=16, hidden_dim=512, n_layers=3, heads=8, dropout=0.2565):
        super().__init__() 
        # Decoder hyperparameters
        self.patch_size = patch_size
        self.hidden_dim = hidden_dim
        self.n_layers = n_layers
        self.heads = heads
        self.dropout = dropout

        # Positional embedding
        self.positional_embedding = PositionalEncoding(hidden_dim= self.hidden_dim)

        # Decoder parameters
        self.decoder_projection = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Dropout(dropout)
        )
        self.decoder_layers = nn.ModuleList([
            nn.Sequential(
                Residual(PreNorm(hidden_dim, Attention(hidden_dim, n_heads=heads, dropout=dropout))),
                Residual(PreNorm(hidden_dim, FeedForward(hidden_dim, hidden_dim, dropout=dropout)))
            )
            for _ in range(n_layers)
        ])
        self.decoder_output_head = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.GELU(),
            nn.Linear(hidden_dim // 2, patch_size),
            nn.Flatten(start_dim=1),
        )

        self._initialize_weights()

    def _initialize_weights(self):
        # Using Xavier uniform initialization for linear layers
        for layer in self.decoder_layers:
            for sublayer in layer:
                if isinstance(sublayer, nn.Linear):
                    nn.init.xavier_uniform_(sublayer.weight)
                    if sublayer.bias is not None:
                        nn.init.zeros_(sublayer.bias)
    
    def forward(self, x):
        """Decodes the encoded tokens using the decoder."""
        # Add positional embedding
        x = self.positional_embedding(x)
        # Decoder projection
        x = self.decoder_projection(x)
        # Decoder layers
        for layer in self.decoder_layers:
            x = layer(x)
        # Remove cls token
        x = x[:, 1:]  
        # Decoder output
        decoded_tokens = self.decoder_output_head(x)

        return decoded_tokens