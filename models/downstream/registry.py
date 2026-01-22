import torch
import numpy as np

from models.downstream.base import DownstreamModel
from models.downstream.classification import LinearClassificationHead, MLPClassificationHead, OldClassificationHead
from models.downstream.regression import LinearRegressionHead,MLPRegressionHead

def get_downstream_model(
    backbone,
    task: str,
    head_type: str,
    classes: np.array,
    freeze_backbone: bool = False,
    device:torch.device | str = "cpu",
):
    """
    Assemble backbone + downstream head.
    """
    assert head_type in ["linear", "non-linear", "old"]
    assert task in ["classification", "regression"]

    if not hasattr(backbone, "hidden_dim"):
        raise AttributeError("Backbone must expose `hidden_dim`")

    in_dim = backbone.hidden_dim
    head_dropout = backbone.dropout
    
    n_classes = len(classes)
    # -------------------------------------------------
    # Head selection
    # -------------------------------------------------
    if task == "classification":

        if head_type == "linear":
            head = LinearClassificationHead(in_dim, n_classes, device)

        elif head_type == "non-linear":
            head = MLPClassificationHead(
                in_dim=in_dim,
                n_classes=n_classes,
                dropout=head_dropout,
                device=device
            )
        elif head_type == "old":
            head = OldClassificationHead(
                in_dim=in_dim,
                n_classes=n_classes,
                dropout=head_dropout,
                device=device
            )

        else:
            raise ValueError(f"Unknown head_type: {head_type}")

    elif task == "regression":

        if head_type == "linear":
            head = LinearRegressionHead(in_dim, device)

        elif head_type == "non-linear":
            head = MLPRegressionHead(
                in_dim=in_dim,
                hidden_dim=in_dim//2,
                dropout=head_dropout,
                device=device,
            )
        else:
            raise ValueError(f"Unknown head_type: {head_type}")

    else:
        raise ValueError(f"Unknown task: {task}")
    
    # Fit the one hot encoding
    head.fit_labelizer(classes)

    # -------------------------------------------------
    return DownstreamModel(
        backbone=backbone,
        head=head,
        freeze_backbone=freeze_backbone,
        device=device,
    )
