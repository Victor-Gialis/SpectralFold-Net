import csv,os, json
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

from pathlib import Path
from tqdm import tqdm
from datetime import datetime

# Utilities for pretraining
def create_run_dir(method:str, dataset:str)->Path:
    """
    Create a directory to save training results.
    Args:
        method (str): Name of the pretraining method.
        dataset (str): Name of the dataset.
    Returns:
        run_dir (Path): Path to the created directory.
    """
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = Path("results")/"pretrain"/f"{method}/{dataset}/{timestamp}"
    run_dir.mkdir(parents=True, exist_ok=True)
    return run_dir

def log_metrics(run_dir:Path, epoch:int, train_loss:float, valid_loss:float):
    """
    Log training and validation metrics to a CSV file.
    Args:
        run_dir (Path): Directory where metrics are logged.
        epoch (int): Current epoch number.
        train_loss (float): Training loss.
        valid_loss (float): Validation loss.
    """
    csv_path = run_dir / "log" / "metrics.csv"
    os.makedirs(csv_path.parent, exist_ok=True)
    file_exists = csv_path.exists() # Check if file already exists
    with open(csv_path, mode='a', newline='') as file:
        writer = csv.writer(file)
        if not file_exists:
            writer.writerow(['epoch', 'train_loss', 'valid_loss'])
        writer.writerow([epoch, train_loss, valid_loss])

def plot_metrics(run_dir:Path):
    """
    Plot training and validation loss curves.
    Args:
        run_dir (Path): Directory where metrics are logged.
    """
    csv_path = run_dir / "log" / "metrics.csv"
    if not csv_path.exists():
        print("No metrics to plot.")
        return

    df = pd.read_csv(csv_path)
    plt.figure()
    plt.plot(df['epoch'], df['train_loss'], label='Train Loss')
    plt.plot(df['epoch'], df['valid_loss'], label='Valid Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training and Validation Loss')
    plt.legend()
    plt.grid()
    plt.savefig(run_dir / "log" / "loss_plot.png")
    plt.close()

def save_model_config(run_dir:Path, 
                      args_dataloader:object, 
                      args_backbone:object, 
                      args_ssl:object, 
                      args_training:object):
    """
    Save model configuration in a json file
    Args:
       run_dir (Path): Directory where model configuration is saved.
       args_dataloader (object): Dataloader configuration
       args_backbone (object): Backbone configuration
       args_ssl (object): SSL configuration
       args_training (object): Training configuration
    """
    config = {
        "dataset":vars(args_dataloader).copy(),
        "backbone":vars(args_backbone).copy(),
        "ssl":vars(args_ssl).copy(),
        "training":vars(args_training).copy(),
    }

    with open(run_dir / "config.json", "w") as f:
        json.dump(config, f, indent=4)

def save_model_checkpoint(run_dir:Path, model:nn.Module, name:str="model.pt"):
    """
    Save model checkpoint.
    Args:
        run_dir (Path): Directory to save the checkpoint.
        model (nn.Module): The model to be saved.
        name (str): Name of the checkpoint file.
    """
    checkpoint_path = run_dir / "checkpoints"
    checkpoint_path.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), checkpoint_path / name)

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

def evaluate(run_dir:Path,
            device:torch.device, 
            model:nn.Module, 
            test_loader:torch.utils.data.DataLoader
            ):
    """
    Evaluation loop for self-supervised learning models.
    Args:
        run_dir (Path): Directory to save logs and visualizations.
        device (torch.device): Device to run the evaluation on.
        model (nn.Module): The model to be evaluated.
        test_loader (DataLoader): DataLoader for test data.
    """
    model.eval()
    test_loss = 0.0

    with torch.no_grad():
        for batch in tqdm(test_loader, desc="Evaluating", leave=False):
            inputs = move_batch_to_device(batch, device)
            outputs = model(inputs)
            
            loss = model.compute_loss(outputs, inputs)
            test_loss += loss.item()

    test_loss /= len(test_loader.dataset)   
    print(f"Test Loss: {test_loss:.4f}")

    # Visulisation of random batch tensor
    for batch in test_loader:
        inputs = move_batch_to_device(batch, device)
        outputs = model(inputs)

        # Get raw and predicted tensors
        x_raw = batch['X_raw'].detach().cpu().numpy()
        x_pred = outputs['prediction'].detach().cpu().numpy()

        # Select random sample from batch
        idx_sample = np.random.randint(0, x_raw.shape[0]-1)

        break # Only need one batch for visualization

    plt.figure(figsize=(12, 6))
    plt.plot(x_raw[idx_sample], label='Raw Signal', alpha=0.7)
    plt.plot(x_pred[idx_sample], label='Reconstructed Signal', alpha=0.7)
    plt.title(f'Signal Reconstruction | Loss: {test_loss:.4f}')
    plt.xlabel('Frequency bins')
    plt.ylabel('Amplitude')
    plt.legend()
    plt.grid()
    plt.savefig(run_dir / "log" / "reconstruction.png")

# Core pretraining loop
def train(
        model:nn.Module,
        train_loader:torch.utils.data.DataLoader,
        valid_loader:torch.utils.data.DataLoader,
        test_loader:torch.utils.data.DataLoader,
        device:torch.device,
        epochs:int,
        optimizer:torch.optim.Optimizer,
        args_dataloader:dict,
        args_backbone:dict,
        args_ssl:dict,
        args_training:dict,
        scheduler:None,
        evaluation:bool=True,
        ):
    """
    Pretraining loop for self-supervised learning models.
    Args:
        model (nn.Module): The model to be trained.
        train_loader (DataLoader): DataLoader for training data.
        valid_loader (DataLoader): DataLoader for validation data.
        test_loader (DataLoader): DataLoader for test data.
        device (torch.device): Device to run the training on.
        epochs (int): Number of epochs to train.
        optimizer (Optimizer): Optimizer for training.
        scheduler (Scheduler or None): Learning rate scheduler.
        evaluation (bool): Whether to evaluate the model after training.
        args_dataloader (dict): Arguments to pass to the dataloader.
        args_backbone (dict): Arguments to pass to the backbone.
        args_ssl (dict): Arguments to pass to the ssl method.
        args_training (dict): Arguments to pass to the training loop.
    Returns:
        None
    """
    # Create run directory
    run_dir = create_run_dir(method=model.__class__.__name__, dataset=train_loader.dataset.dataset.__class__.__name__) 
    
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

        if run_dir:
            log_metrics(run_dir, epoch, train_loss, valid_loss)

        # Save best model
        if run_dir and (epoch == 1 or valid_loss < best_valid_loss):
            best_valid_loss = valid_loss
            save_model_checkpoint(run_dir, model, name="best.pt")
    
    # Save final model
    if run_dir:
        save_model_checkpoint(run_dir, model, name="last.pt")
        save_model_config(run_dir, args_dataloader, args_backbone, args_ssl, args_training) # Save config file
        plot_metrics(run_dir) # Plot training metrics

    print("Training complete.")

    # Evaluate on test set
    if evaluation:
        evaluate(
            run_dir=run_dir,
            device=device,
            model=model,
            test_loader=test_loader,
        )