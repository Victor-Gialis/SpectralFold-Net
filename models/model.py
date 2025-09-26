import torch
import torch.nn as nn
from utils.transformer_blocks import PatchEmbedding, Attention, PreNorm, FeedForward, Residual, _sincos_pos_enc

class Encoder(nn.Module):
    def __init__(self, num_patch=16, patch_size=64, encoder_dim=128, n_layers=6, heads=8, dropout=0.4):
        super().__init__()
        # Encoder hyperparameters
        self.num_patch = num_patch
        self.patch_size = patch_size
        self.encoder_dim = encoder_dim
        self.n_layers = n_layers
        self.heads = heads
        self.dropout = dropout

        # Positional embedding
        self.positional_embedding = _sincos_pos_enc(num_patch= self.num_patch,
                                                    encoder_dim= self.encoder_dim)

        # Patch embedding
        self.patch_embedding = PatchEmbedding(patch_size=patch_size, emb_size=encoder_dim)

        # Class token
        self.cls_token = nn.Parameter(torch.zeros(1, 1, encoder_dim))

        # Encoder parameters
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
            nn.ReLU(),
            nn.Dropout(dropout)
        )
    
    def _initialize_weights(self):
        # Initialize patch embedding weights
        nn.init.xavier_uniform_(self.patch_embedding.projection[-1].weight)
        if self.patch_embedding.projection[-1].bias is not None:
            nn.init.zeros_(self.patch_embedding.projection[-1].bias)

        # Initialize cls token
        nn.init.normal_(self.cls_token, std=0.02)

        # using Xavier uniform initialization for linear layers
        for layer in self.encoder_layers:
            for sublayer in layer:
                if isinstance(sublayer, nn.Linear):
                    nn.init.xavier_uniform_(sublayer.weight)
                    if sublayer.bias is not None:
                        nn.init.zeros_(sublayer.bias)
    
    def forward(self, x):
        """Encodes the input using the encoder."""
        # Patch embedding
        x = self.patch_embedding(x)
        # Extract batch size and number of tokens
        b, n, _ = x.shape

        # Add cls token
        cls_tokens = self.cls_token.expand(b, -1, -1)  # Étendre le cls_token pour correspondre à la taille du batch
        x = torch.cat([cls_tokens, x], dim=1)  # Concaténer le cls_token avec les embeddings des patches
        # Add positional embedding
        x = x + self.positional_embedding[:,:n+1]

        # Encoder layers
        for layer in self.encoder_layers:
            x = layer(x)

        # Encoder output
        encoded_tokens = self.encoder_output_head(x)
        return encoded_tokens
        
class Decoder(nn.Module):
    def __init__(self, num_patch=16, patch_size=64, encoder_dim=128, decoder_dim=256, n_layers=6, heads=8, dropout=0.4):
        super().__init__() 
        # Decoder hyperparameters
        self.num_patch = num_patch
        self.patch_size = patch_size
        self.encoder_dim = encoder_dim
        self.decoder_dim = decoder_dim
        self.n_layers = n_layers
        self.heads = heads
        self.dropout = dropout

        # Positional embedding
        self.positional_embedding = _sincos_pos_enc(num_patch= self.num_patch,
                                                    encoder_dim= self.encoder_dim)

        # Mask token
        self.mask_token = nn.Parameter(torch.zeros(1, 1, encoder_dim))

        # Decoder parameters
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
            nn.Linear(decoder_dim, decoder_dim),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(decoder_dim, decoder_dim // 2),
            nn.GELU(),
            nn.Linear(decoder_dim // 2, patch_size),
            nn.Flatten(start_dim=1),
        )
    
    def _initialize_weights(self):
        # Iniatialize mask token
        nn.init.normal_(self.mask_token, std=0.02)

        # using Xavier uniform initialization for linear layers
        for layer in self.decoder_layers:
            for sublayer in layer:
                if isinstance(sublayer, nn.Linear):
                    nn.init.xavier_uniform_(sublayer.weight)
                    if sublayer.bias is not None:
                        nn.init.zeros_(sublayer.bias)
    
    def forward(self, x):
        """Decodes the encoded tokens using the decoder."""
        # Extract batch size and number of tokens
        b, n, _ = x.shape
        # Padding patch tokens using mask token
        if n < self.num_patch:
            bottom_padding = self.mask_token.expand(b, self.num_patch - n +1, -1)  # Create padding tokens
            bottom_padding = bottom_padding.to(x.device)  # Move to the same device as x
            x = torch.cat([x, bottom_padding], dim=1)  # bottom padding

        # Add positional embedding
        x = x + self.positional_embedding
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
class PretrainedModel(nn.Module):
    def __init__(self, input_size:int, patch_size=64, encoder_dim=128, decoder_dim=256, n_layers=6, heads=8, dropout=0.4):
        super().__init__()

        # Save hyperparameters
        self.input_size = input_size
        self.patch_size = patch_size
        self.encoder_dim = encoder_dim
        self.decoder_dim = decoder_dim
        self.n_layers = n_layers
        self.heads = heads
        self.dropout = dropout
        self.num_patch = input_size // patch_size

        # Create encoder object
        self.encoder = Encoder(
            num_patch= self.num_patch,
            patch_size= self.patch_size,
            encoder_dim= self.encoder_dim,
            n_layers= self.n_layers,
            heads= self.heads,
            dropout= self.dropout
        )

        # Create decoder object
        self.decoder = Decoder(
            num_patch= self.num_patch,
            patch_size= self.patch_size,
            encoder_dim= self.encoder_dim,
            decoder_dim= self.decoder_dim,
            n_layers= self.n_layers,
            heads= self.heads,
            dropout= self.dropout           
        )
        
    def forward(self, x):
        # Forward through encoder
        encoded_tokens = self.encoder(x)
        # Forward through decoder
        decoded_tokens = self.decoder(encoded_tokens)

        return decoded_tokens

class DownstreamClassifier(nn.Module):
    def __init__(self, backbone, num_classes=4,freeze_backbone=False):
        super().__init__()
        self.backbone = backbone
        backbone_dim = backbone.encoder_dim
        if freeze_backbone:
            for param in self.backbone.parameters():
                param.requires_grad = False
                
        self.classifier = nn.Sequential(
            nn.LayerNorm(backbone_dim ),
            nn.Linear(backbone_dim , backbone_dim // 2),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(backbone_dim // 2, num_classes)
        )
        
    def forward(self, x):
        backbone_tokens = self.backbone(x)
        cls_token = backbone_tokens[:, 0]
        logits = self.classifier(cls_token)
        return logits