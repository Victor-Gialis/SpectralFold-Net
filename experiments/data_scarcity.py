import torch
import random
import argparse
import numpy as np
from tqdm import tqdm
from types import SimpleNamespace

from dataset import dataloader
from training.downstream import train, evaluate, log_metrics
from models.downstream.registry import get_downstream_model
from models.backbone.registry import get_backbone
from models.ssl.registry import get_pretrained_backbone

def main(args):
    """
    Run ONE data-scarcity downstream experiment.
    Args:
        args (object): Arguments for the experiment.
    """

    # Set seed for reproducibility
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)

    # Device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Dataloader configuration
    args_dataloader = SimpleNamespace(
        name=args.downstream_dataset,
        window_size=args.window_size,
        window_stride=args.window_stride,
        batch_size=args.batch_size,
        data_ratio=args.data_ratio,
    )
    
    train_loader, valid_loader, test_loader, labels = dataloader.get_homogenous_split_dataloaders(args_dataloader)

    # Backbone
    if args.backbone_init == "random":
        args.pretrain_dataset = None  # No pretraining dataset needed for random initialization

        # Set backbone arguments
        args_backbone = SimpleNamespace(
            model="vit1d",
        )
        backbone = get_backbone(args_backbone)

        # Compute stats from X_raw train dataloader
        stats = dataloader.compute_stats_from_dataloader(train_loader)
        backbone._loads_stats(stats)
    
    else:
        backbone = get_pretrained_backbone(
            method=args.backbone_init,
            pretrain_dataset=args.pretrain_dataset,
            device=device,
        )

    # Downstream model
    model = get_downstream_model(
        backbone=backbone,
        task=args.task, 
        head_type=args.head_type,# "classification" | "regression"
        classes=labels,
        freeze_backbone= not args.finetune,
        device=device
    )

    # Optimisation
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.learning_rate,
        weight_decay=args.weight_decay,
    )

    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=args.epochs,
    )

    # Training
    train(
        args=args,
        model=model,
        train_loader=train_loader,
        valid_loader=valid_loader,
        test_loader=test_loader,
        device=device,
        optimizer=optimizer,
        scheduler=scheduler,
        epochs=args.epochs,
    )

if __name__ == "__main__":
    # parser = argparse.ArgumentParser("Downstream data scarcity experiment")

    # # SSL
    # parser.add_argument("--backbone_init", type=str, choices=["random","sap", "mae"], required=True) # Require: sap | mae
    # parser.add_argument("--pretrain_dataset", type=str, default="CWRU")

    # # Downstream dataset
    # parser.add_argument("--downstream_dataset", type=str, choices=["CWRU", "LASPI"], required=True) 
    # parser.add_argument("--data_ratio", type=float, default=1.0)

    # # Probing
    # parser.add_argument("--head_type", type=str, choices=["linear", "non-linear"], required=True)
    # parser.add_argument("--finetune", action="store_true")

    # # Task
    # parser.add_argument("--task", type=str, choices=["classification", "regression"], default="classification")

    # # Training
    # parser.add_argument("--seed", type=int, required=True)
    # parser.add_argument("--epochs", type=int, required=True)
    # parser.add_argument("--batch_size", type=int, default=64)
    # parser.add_argument("--learning_rate", type=float, default=0.0003695)
    # parser.add_argument("--weight_decay", type=float, default=1.1133e-5)

    # # Signal
    # parser.add_argument("--window_size", type=int, default=2048)
    # parser.add_argument("--window_stride", type=int, default=256)

    # args = parser.parse_args()
    # main(args)

    # Debug example
    args = SimpleNamespace(
        backbone_init="random",  # "random" | "sap" | "mae"
        pretrain_dataset="CWRU",
        downstream_dataset="CWRU",
        data_ratio=0.01,
        head_type="linear",
        finetune=True,
        task="classification",
        seed=2,
        epochs=20,
        batch_size=64,
        learning_rate=0.0003695,
        weight_decay=1.1133e-5,
        window_size=2048,
        window_stride=256,
    )
    main(args)