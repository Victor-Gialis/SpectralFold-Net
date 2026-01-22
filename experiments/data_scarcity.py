import torch
import argparse
from tqdm import tqdm
from types import SimpleNamespace

from dataset import dataloader
from training.downstream import train, evaluate, save_results_csv
from models.downstream.registry import get_downstream_model
from models.backbone.registry import get_backbone
from models.ssl.registry import get_pretrained_backbone

def main(args):
    """
    Run ONE data-scarcity downstream experiment.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Dataloader configuration
    args_dataloader = SimpleNamespace(
        name=args.dataset,
        window_size=args.window_size,
        window_stride=args.window_stride,
        batch_size=args.batch_size,
        data_ratio=args.data_ratio,
    )
    
    train_loader, valid_loader, test_loader, labels = dataloader.get_homogenous_split_dataloaders(args_dataloader)

    # Backbone
    if args.ssl_method == "random":
        print("je prend le random")
        # Set backbone arguments
        args_backbone = SimpleNamespace(
            model="vit1d",
        )
        backbone = get_backbone(args_backbone)

        # Compute min-max from X_raw train dataloader
        stats = dataloader.compute_min_max_from_dataloader(train_loader)
        backbone._loads_stats(stats)
    
    else:
        print("je prend le pretrain")
        backbone = get_pretrained_backbone(
            method=args.ssl_method,
            pretrain_dataset=args.pretrain_dataset,
            device=device,
        )

    backbone.to(device)

    # Downstream model
    model = get_downstream_model(
        backbone=backbone,
        task=args.task, 
        head_type=args.head_type,# "classification" | "regression"
        classes=labels,
        freeze_backbone= not args.finetune,
        device=device
    ).to(device)

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

    print("Training on; Dataset:{}; SSL:{}; Head:{}; Seed:{}".format(args.dataset, args.ssl_method, args.head_type, args.seed))

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

    # Évaluation
    metrics = evaluate(model, test_loader, device, task=args.task)

    # Préparer le dictionnaire pour le CSV
    results_dict = {
        "pretrain_dataset":args.pretrain_dataset,
        "downstream_dataset": args.dataset,
        "epochs":args.epochs,
        "ssl_method": args.ssl_method,
        "head_type": args.head_type,
        "finetune":args.finetune,
        "seed": args.seed,
        "data_ratio": args.data_ratio,
    }
    results_dict.update(metrics)

    # Sauvegarde dans CSV
    save_results_csv(results_dict)
    print(f"Results added to csv : {results_dict}")

if __name__ == "__main__":
    # parser = argparse.ArgumentParser("Downstream data scarcity experiment")

    # # SSL
    # parser.add_argument("--ssl_method", type=str, choices=["random","sap", "mae"], required=True) # Require: sap | mae
    # parser.add_argument("--pretrain_dataset", type=str, default="CWRU")

    # # Downstream dataset
    # parser.add_argument("--dataset", type=str, choices=["CWRU", "LASPI"], required=True) 
    # parser.add_argument("--data_ratio", type=float, default=1.0)

    # # Probing
    # parser.add_argument("--head_type", type=str, choices=["linear", "non-linear"], required=True)
    # parser.add_argument("--finetune", action="store_true")

    # # Task
    # parser.add_argument("--task", type=str, choices=["classification", "regression"], default="classification")

    # # Training
    # parser.add_argument("--seed", type=int, required=True)
    # parser.add_argument("--epochs", type=int, default=50)
    # parser.add_argument("--batch_size", type=int, default=64)
    # parser.add_argument("--learning_rate", type=float, default=0.0003695)
    # parser.add_argument("--weight_decay", type=float, default=1.1133e-5)

    # # Signal
    # parser.add_argument("--window_size", type=int, default=2048)
    # parser.add_argument("--window_stride", type=int, default=256)

    # args = parser.parse_args()
    # main(args)

    from types import SimpleNamespace

    # Args fixés pour debug local
    args = SimpleNamespace(
        # SSL
        ssl_method="random",          # "sap" ou "mae"
        pretrain_dataset="CWRU",

        # Downstream dataset
        dataset="CWRU",
        data_ratio=0.01,           # petit ratio pour debug

        # Probing
        head_type="linear",        # "linear" ou "non-linear"
        finetune=True,            # True si tu veux tester le finetune

        # Task
        task="classification",     # "classification" ou "regression"

        # Training
        seed=20,
        epochs=100,                  # petit nombre d'epochs pour debug
        batch_size=64,             # batch réduit pour PC
        learning_rate=0.0003695,
        weight_decay=1.1133e-5,

        # Signal
        window_size=2048,
        window_stride=256,
    )

    main(args)
