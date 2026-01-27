import torch
import torch.nn as nn
from dataset.transform import normalization

class DownstreamModel(nn.Module):
    def __init__(
        self,
        backbone: nn.Module,
        head: nn.Module,
        freeze_backbone: bool = False,
        device:  torch.device | str = "cpu",
    ):
        super().__init__()

        self.backbone = backbone
        self.head = head

        if freeze_backbone:
            for p in self.backbone.parameters():
                p.requires_grad = False

    def forward(self, batch):
        """
        batch: dict (standardisé par ton dataloader)
        """
        x_raw = batch['X_raw']
        
        # Normalization
        # x_norm = normalization.global_z_log_normalization(x=x_raw, stats=self.backbone.stats)
        x_norm = normalization.global_min_max_log_normalization(x=x_raw, stats=self.backbone.stats)

        features = self.backbone(x_norm)
        cls_token = features[:,0]
        outputs = self.head(cls_token)
        
        return outputs

    def compute_loss(self, outputs, batch):
        return self.head.compute_loss(outputs, batch)
