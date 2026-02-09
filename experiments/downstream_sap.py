import torch, os
import random
import argparse
import numpy as np
from tqdm import tqdm
from types import SimpleNamespace

from dataset import dataloader
from training.downstream import train, evaluate, log_metrics
from training.pretrain import load_model_checkpoint
from models.downstream.registry import get_downstream_model
from models.backbone.vit1d import ViT1DEncoder
from models.backbone.registry import get_backbone
from models.ssl.registry import get_pretrained_backbone
from models.ssl.sap import SAPModel

def main(args):
    """
    Run ONE data-scarcity downstream experiment.
    Args:
        args (object): Arguments for the experiment.
    """

    # Set seed for reproducibility
    torch.manual_seed(42)
    np.random.seed(42)
    random.seed(42)

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
    
    # # Split dataloaders - sampling spectrum independently
    # train_loader, valid_loader, test_loader, labels = dataloader.get_homogenous_split_dataloaders(args_dataloader,seed=args.seed)

    # Split dataloaders - speed stratified sampling
    if args.downstream_dataset == "CWRU":
        train_loader, valid_loader, test_loader, labels = dataloader.get_speed_stratified_dataloaders(
            args=args_dataloader,
            train_val_speeds=['1750', '1772', '1797'],
            test_speeds=['1730'],
            seed=args.seed,
        )
    
    # elif args.downstream_dataset == "LASPI":
        ## Original speed stratified
        # train_loader, valid_loader, test_loader, labels = dataloader.get_speed_stratified_dataloaders(
        #     args=args_dataloader,
        #     train_val_speeds=[25, 45],
        #     test_speeds=[35],
        #     seed=args.seed,
        # )

        # # Mixed speed train/val, test on fixed speeds and load conditions
        # train_loader, valid_loader, test_loader, labels = dataloader.get_speed_load_stratified_dataloaders(
        #     args=args_dataloader, 
        #     train_val_combinations=[(25, 0), (25, 25), (25, 75),
        #                             (35, 0), (35, 25), (35, 50), (35, 75),
        #                             (45, 0), (45, 25), (45, 50), (45, 75),],
        #     test_combinations=[(25, 50)],
        #     seed=args.seed,
        # )

        # # Mixed speed train/val, test on fixed speeds and load conditions, but only 1 load condition per speed to avoid overfitting to load conditions
        # train_loader, valid_loader, test_loader, labels = dataloader.get_laspi_acquisition_split_dataloaders(
        #     args=args_dataloader,
        #     seed=args.seed,
        # )


    # Backbone
    backbone_random = ViT1DEncoder()
    ssl_mode = SAPModel(backbone=backbone_random,
                        args_ssl=dict()
                        )
    
    # get last checkpoint for MAE pretrained on CWRU
    checkpoints = os.listdir("results/pretrain/SAPModel/CWRUDataset")
    checkpoints.sort()

    print("Available checkpoints:", checkpoints)
    checkpoint_path = os.path.join("results/pretrain/SAPModel/CWRUDataset", checkpoints[-1])  # Get the last checkpoint

    # checkpoint_path = "results/pretrain/SAPModel/CWRUDataset/20260123_165739"
    ssl_model = load_model_checkpoint(ssl_mode, checkpoint_path)
    
    backbone = ssl_model.backbone

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

    # Learning rate scheduler
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
    # Debug example
    for data_ratio in [1.0]:
        for fin in [True, False]:
            for seed in range(1):
                args = SimpleNamespace(
                    backbone_init="sap",  # "random" | "sap" | "mae"
                    pretrain_dataset="CWRU",
                    downstream_dataset="CWRU",
                    data_ratio=data_ratio,
                    head_type="linear",
                    finetune=fin,
                    task="classification",
                    seed=seed,
                    epochs=50,
                    batch_size=64,
                    learning_rate=0.0003695,
                    weight_decay=1.1133e-5,
                    window_size=2048,
                    window_stride=256,
                )
                main(args)