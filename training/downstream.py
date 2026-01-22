import csv,os
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

from pathlib import Path
from tqdm import tqdm
from datetime import datetime

from sklearn.metrics import accuracy_score, f1_score, mean_squared_error, r2_score

def evaluate(model, test_loader, device, task="classification"):
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
    else:
        metrics["mse"] = mean_squared_error(all_labels, all_preds)
        metrics["r2"] = r2_score(all_labels, all_preds)

    return metrics

def save_results_csv(results_dict, csv_path="results/downstream/results_summary.csv"):
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
    
    # Move model to device
    model.to(device)
    best_valid_loss = float('inf')

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
        # train_loss /= len(train_loader.dataset)

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
        # valid_loss /= len(valid_loader.dataset)

        if scheduler:
            scheduler.step()

        print(f"Epoch {epoch}/{epochs} - Train Loss: {train_loss:.4f}, Valid Loss: {valid_loss:.4f}")



