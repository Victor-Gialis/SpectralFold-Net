import torch
import torch.nn as nn

from dataset.transform import normalization

from models.backbone.vit1d import ViT1DDecoder
from models.ssl.base import BaseSSLModel

class SAPModel(BaseSSLModel):
    def __init__(self, backbone:nn.Module, args_ssl:dict):
        """
        Spectral Aliasing Pretext (SAP) model for self-supervised learning on time series data.
        Args:
            backbone (nn.Module): Backbone model to extract features from.
            args_ssl (dict): Arguments for self-supervised model.
        """
        super(BaseSSLModel, self).__init__()

        # Backbone model to extract features from
        self.backbone = backbone
        self.args_ssl = args_ssl

        # Downsampling factor for SAP
        self.downsample_factor = args_ssl.downsampling_factor if hasattr(args_ssl, 'downsampling_factor') else 1

        # Create decoder
        self.decoder = ViT1DDecoder()
        
        # Loss function
        self.loss_function = torch.nn.MSELoss()

        # Initialize weights
        self._initialize_weights()
    
    def _initialize_weights(self):
        # Initialize weights if necessary
        pass

    def forward(self, batch):
        """
        Forward pass of the SAP model.
        Args:
            batc (torch.Tensor): Input batch of shape (batch_size, seq_length).
        Returns:
        """
        x_raw = batch['X_raw'] # get raw spectre without fold
        x_fold = batch['X_folded'] # get folded spectrum in input
        
        # Normalization
        x_norm = normalization.global_z_log_normalization(x=x_fold, stats=self.backbone.stats)
        
        # Forward pass
        embedded_tokens = self.backbone(x_norm)
        x_pred_norm = self.decoder(embedded_tokens)
        
        # Unnormalize
        x_pred = normalization.global_z_log_unnormalization(x_norm=x_pred_norm, stats=self.backbone.stats)

        # Ensure non-negative outputs
        x_pred = super().non_negative_output(x_pred)

        return {"prediction": x_pred}
    
    def compute_loss(self, outputs, inputs):
        targets = inputs['X_raw']
        predictions = outputs['prediction']
        return self.loss_function(predictions, targets)