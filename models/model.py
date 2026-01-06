import torch
import torch.nn as nn
from utils.transformer_blocks import Tokeniser, Attention, PreNorm, FeedForward, Residual, PositionalEncoding

class Encoder(nn.Module):
    def __init__(self, patch_size=64, hidden_dim=128, n_layers=6, heads=8, dropout=0.4):
        super().__init__()
        # Encoder hyperparameters
        self.patch_size = patch_size
        self.hidden_dim = hidden_dim
        self.n_layers = n_layers
        self.heads = heads
        self.dropout = dropout

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
    
    def forward(self, x):
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
        
class Decoder(nn.Module):
    def __init__(self, patch_size=64, hidden_dim=128, n_layers=6, heads=8, dropout=0.4):
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
class PretrainingModel(nn.Module):
    def __init__(self, pretext_task='SAP', patch_size=64, hidden_dim=128, n_layers=6, heads=8, dropout=0.4):
        super().__init__()
        assert pretext_task in ['SAP','MAE']

        # Save hyperparameters
        self.pretext_task = pretext_task
        self.patch_size = patch_size
        self.hidden_dim = hidden_dim
        self.n_layers = n_layers
        self.heads = heads
        self.dropout = dropout

        # Mask token
        self.mask_token = nn.Parameter(torch.zeros(1, 1, hidden_dim))

        # Create tokeniser object
        self.tokeniser = Tokeniser(patch_size=patch_size, emb_size=hidden_dim)

        # Create encoder object
        self.encoder = Encoder(
            patch_size= self.patch_size,
            hidden_dim= self.hidden_dim,
            n_layers= self.n_layers,
            heads= self.heads,
            dropout= self.dropout
        )

        # Create decoder object
        self.decoder = Decoder(
            patch_size= self.patch_size,
            hidden_dim= self.hidden_dim,
            n_layers= self.n_layers,
            heads= self.heads,
            dropout= self.dropout           
        )

        self._initialize_weights()

    
    def _initialize_weights(self):
        # Iniatialize mask token
        nn.init.normal_(self.mask_token, std=0.02)

        # Initialize patch embedding weights
        nn.init.xavier_uniform_(self.tokeniser.projection[-1].weight)

        if self.tokeniser.projection[-1].bias is not None:
            nn.init.zeros_(self.tokeniser.projection[-1].bias)
        
    def forward(self, x):
        # Forward through tokeniser
        input_tokens = self.tokeniser(x)

        # Masked token
        if self.pretext_task == 'MAE':
            input_tokens, mask, ids_restore = self.random_masking(input_tokens, mask_ratio=0.05)

        # Forward through encoder
        embedded_tokens = self.encoder(input_tokens)

        # Append mask tokens to sequence
        if self.pretext_task == 'MAE':
            b, n, d = embedded_tokens.shape
            mask_tokens = self.mask_token.repeat(b, ids_restore.shape[1] + 1 - n, 1)
            embedded_tokens_ = torch.cat([embedded_tokens[:, 1:, :], mask_tokens], dim=1)  # no cls token
            embedded_tokens_ = torch.gather(embedded_tokens_, dim=1, index=ids_restore.unsqueeze(-1).repeat(1, 1, d))  # unshuffle
            embedded_tokens_ = torch.cat([embedded_tokens[:, :1, :], embedded_tokens_], dim=1)  # append cls token

        # Forward through decoder
        output_tokens = self.decoder(embedded_tokens_)

        return output_tokens, mask
    
    def random_masking(self, tokens, mask_ratio):
        """Performs random masking on the input tokens."""
        b, n, d = tokens.shape
        len_keep = int(n * (1 - mask_ratio))
        noise = torch.rand(b, n, device=tokens.device)

        ids_shuffle = torch.argsort(noise, dim=1)
        ids_restore = torch.argsort(ids_shuffle, dim=1)
        ids_keep = ids_shuffle[:, :len_keep]
        masked_tokens = tokens.gather(dim=1, index=ids_keep.unsqueeze(-1).repeat(1, 1, d))
        
        mask = torch.ones([b, n], device=tokens.device)
        mask[:, :len_keep] = 0
        mask = torch.gather(mask, dim=1, index=ids_restore)
        
        return masked_tokens, mask, ids_restore

class DownstreamClassifier(nn.Module):
    def __init__(self, backbone: PretrainingModel, num_classes=4,freeze_backbone=False):
        super().__init__()
        self.backbone = backbone

        self.tokeniser = self.backbone.tokeniser
        self.encoder = self.backbone.encoder
        self.hidden_dim = self.backbone.hidden_dim

        if freeze_backbone:
            for param in self.backbone.parameters():
                param.requires_grad = False
        
        # Non-linear probing classifier head
        self.classifier = nn.Sequential(
            nn.LayerNorm(self.hidden_dim),
            nn.Linear(self.hidden_dim, self.hidden_dim// 2),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(self.hidden_dim // 2, num_classes)
        )
        
    def forward(self, x):
        # Forward through tokeniser and encoder
        input_tokens = self.tokeniser(x)
        embedded_tokens = self.encoder(input_tokens)
        # Classification head on cls token
        cls_token = embedded_tokens[:, 0]
        logits = self.classifier(cls_token)
        return logits