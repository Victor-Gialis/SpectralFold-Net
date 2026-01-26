import csv,os
import torch, json
import torch.nn as nn
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import seaborn as sns

from pathlib import Path
from tqdm import tqdm
from datetime import datetime

from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from sklearn.metrics import accuracy_score, f1_score, mean_squared_error, r2_score

def create_run_dir(base_dir="results/downstream", args:object=None):
    """
    Docstring for create_run_dir
    
    :param base_dir: Description
    :param args: Description
    :type args: object
    """ 
    pretrain_dataset = args.pretrain_dataset 
    downstream_dataset = args.downstream_dataset
    backbone_init = args.backbone_init
    head_type = args.head_type
    finetune = args.finetune

    if pretrain_dataset is None:
        pretrain_dataset = "scratch"

    # create directory structure
    dir = f"{pretrain_dataset}_{downstream_dataset}_backbone_{backbone_init}_head_{head_type}_finetune_{finetune}"
    run = os.path.join(base_dir, dir)
    os.makedirs(run, exist_ok=True)

    data_ratio = args.data_ratio
    epochs = args.epochs

    # create sub-directory for data ratio and epochs
    dir_name = f"data_ratio_{data_ratio}_epochs_{epochs}"
    run_dir = os.path.join(run, dir_name)
    os.makedirs(run_dir, exist_ok=True)

    return run_dir

def save_model_config(run_dir, args):
    ### save the model configuration in json format
    config_path = os.path.join(run_dir, "config.json")
    with open(config_path, "w") as f:
        json.dump(vars(args), f, indent=4)
    
def save_model_checkpoint(model, run_dir, name="model_checkpoint.pth"):
    """
    Docstring for save_model_checkpoint
    
    :param model: Description
    :param run_dir: Description
    :param name: Description
    """
    checkpoint_path = os.path.join(run_dir, name)
    torch.save(model.state_dict(), checkpoint_path)

def evaluate(run_dir, model, test_loader, device, task="classification"):
    """
    Docstring for evaluate
    
    :param model: Description
    :param test_loader: Description
    :param device: Description
    :param task: Description
    """
    model.eval()
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for batch in tqdm(test_loader, desc="Evaluating", leave=False):
            inputs = move_batch_to_device(batch, device)
            outputs = model(inputs)

            if task == "classification":
                targets = model.head.transform_labels(inputs['y_label'])
                targets = torch.argmax(targets, dim=1)
                preds = torch.argmax(outputs, dim=1)
            else:  # regression
                preds = outputs.squeeze()
            
            all_preds.append(preds.detach().cpu())
            all_labels.append(targets.detach().cpu())
    
    all_preds = torch.cat(all_preds)
    all_labels = torch.cat(all_labels)

    metrics = {}
    if task == "classification":
        metrics["accuracy"] = accuracy_score(all_labels, all_preds)
        metrics["f1"] = f1_score(all_labels, all_preds, average="weighted")

        # Confusion matrix
        cm = confusion_matrix(all_labels, all_preds)
        disp = ConfusionMatrixDisplay(confusion_matrix=cm)
        disp.plot()
        plt.savefig(os.path.join(run_dir, 'confusion_matrix.png'))
        plt.close()

    else:
        metrics["mse"] = mean_squared_error(all_labels, all_preds)
        metrics["r2"] = r2_score(all_labels, all_preds)

    return metrics

def log_metrics(results_dict, csv_path="results/downstream/results_summary.csv"):
    os.makedirs(os.path.dirname(csv_path), exist_ok=True)
    
    file_exists = os.path.isfile(csv_path)
    
    with open(csv_path, mode="a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=results_dict.keys())
        if not file_exists:
            writer.writeheader()
        writer.writerow(results_dict)

def move_batch_to_device(batch, device:torch.device):
    """
    Move batch data to the specified device.
    Args:
        batch (dict): Batch data containing tensors.
        device (torch.device): Target device.
    Returns:
        dict: Batch data on the target device.
    """
    for k, v in batch.items():
        if torch.is_tensor(v):
            batch[k] = v.to(device, non_blocking=True)
    return batch

def train(
    args:object,
    model:nn.Module,
    train_loader:torch.utils.data.DataLoader,
    valid_loader:torch.utils.data.DataLoader,
    test_loader:torch.utils.data.DataLoader,
    device:torch.device,
    optimizer,
    scheduler,
    epochs,
):    
    # Create run directory
    run_dir = create_run_dir(base_dir="results/downstream", args=args)

    all_train_losses = []
    all_valid_losses = []
    
    # Move model to device
    model.to(device)
    best_valid_loss = float("inf")

    for epoch in range(1, epochs + 1):
        # Training phase
        model.train()
        train_loss = 0.0

        for batch in tqdm(train_loader, desc=f"Epoch {epoch}/{epochs} - Training", leave=False):
            inputs = move_batch_to_device(batch, device)
            optimizer.zero_grad()
            
            outputs = model(inputs)
            loss = model.compute_loss(outputs, inputs)

            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()

        train_loss /= len(train_loader)
        all_train_losses.append(train_loss)

        # Validation phase
        model.eval()
        valid_loss = 0.0

        with torch.no_grad():
            for batch in tqdm(valid_loader, desc=f"Epoch {epoch}/{epochs} - Validation", leave=False):
                inputs = move_batch_to_device(batch, device)
                outputs = model(inputs)
                
                loss = model.compute_loss(outputs, inputs)
                valid_loss += loss.item()

        valid_loss /= len(valid_loader)
        all_valid_losses.append(valid_loss)

        if scheduler:
            scheduler.step()

        # # Save best model
        # if run_dir and (epoch == 1 or valid_loss < best_valid_loss):
        #     best_valid_loss = valid_loss
        #     save_model_checkpoint(model, run_dir, name="best.pt")

        print(f"Epoch {epoch}/{epochs} - Train Loss: {train_loss:.4f}, Valid Loss: {valid_loss:.4f}")

    # # Save final model
    # if run_dir:
    #     save_model_checkpoint(model, run_dir, name="final.pt")
    
    # Testing phase
    metrics = evaluate(run_dir, model, test_loader, device, task=args.task)
    
    # Préparer le dictionnaire pour le CSV
    results_dict = {
        "pretrain_dataset":args.pretrain_dataset,
        "downstream_dataset": args.downstream_dataset,
        "epochs":args.epochs,
        "backbone_init": args.backbone_init,
        "head_type": args.head_type,
        "finetune":args.finetune,
        "seed": args.seed,
        "data_ratio": args.data_ratio,
    }
    results_dict.update(metrics)

    # Sauvegarde de la configuration du modèle
    save_model_config(run_dir, args)

    # Sauvegarde dans CSV
    log_metrics(results_dict)
    print(f"Results added to csv : {results_dict}")