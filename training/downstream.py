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
from models.ssl.mae import MAEModel

from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from sklearn.metrics import accuracy_score, f1_score, mean_squared_error, r2_score
from matplotlib.patches import Rectangle
from dataset.transform import normalization

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
    seed = args.seed

    if backbone_init == "random":
        pretrain_dataset = "None"

    # create directory structure
    dir = f"{pretrain_dataset}_to_{downstream_dataset}_backbone_{backbone_init}_head_{head_type}_finetune_{finetune}"
    run = os.path.join(base_dir, dir)
    os.makedirs(run, exist_ok=True)

    data_ratio = args.data_ratio
    epochs = args.epochs

    # create sub-directory for data ratio and epochs
    dir_name = f"data_ratio_{data_ratio}_epochs_{epochs}_seed_{seed}"
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

def plot_metrics(train_losses, valid_losses, run_dir):
    """
    Docstring for plot_metrics
    
    :param train_losses: Description
    :param valid_losses: Description
    :param run_dir: Description
    """
    plt.figure(figsize=(10, 6))
    plt.plot(train_losses, label="Train Loss")
    plt.plot(valid_losses, label="Validation Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Training and Validation Loss over Epochs")
    plt.legend()
    plt.grid()
    plt.savefig(os.path.join(run_dir, "loss_curve.png"))
    plt.close()

def plot_attention_spectra(run_dir, model, test_loader, device, task="classification"):
    """
    Plot average spectrum for each class with patches colored by attention scores.
    
    :param run_dir: Directory to save the plots
    :param model: Model with backbone that has get_attention_scores()
    :param test_loader: Test data loader
    :param device: Device to run on
    :param task: Task type (classification or regression)
    """
    if task != "classification":
        return
    
    model.eval()
    class_spectra = {}
    class_attention_scores = {}
    
    with torch.no_grad():
        for batch in tqdm(test_loader, desc="Extracting Attention Scores", leave=False):
            inputs = move_batch_to_device(batch, device)
            
            # Get input spectra
            X_raw = inputs['X_raw']  # (batch, n_time_steps)
            
            # Normalize (same as in downstream model)
            x_norm = normalization.global_z_log_normalization(x=X_raw, stats=model.backbone.stats)
            
            # Get attention scores from backbone for each sample
            batch_attention_scores = []
            for i in range(x_norm.shape[0]):
                single_sample = x_norm[i:i+1]  # (1, n_time_steps)
                attn_score = model.backbone.get_attention_scores(single_sample)  # (n_patches,)
                batch_attention_scores.append(attn_score.cpu().numpy())
            
            batch_attention_scores = np.array(batch_attention_scores)  # (batch, n_patches)
            
            # Get forward pass for class labels
            outputs = model(inputs)
            targets = model.head.transform_labels(inputs['y_label'])
            class_labels = torch.argmax(targets, dim=1).cpu().numpy()
            
            # Get input spectra
            X = X_raw.cpu().numpy()  # (batch, n_time_steps)
            
            # Aggregate by class
            for i, class_idx in enumerate(class_labels):
                class_idx = int(class_idx)
                if class_idx not in class_spectra:
                    class_spectra[class_idx] = []
                    class_attention_scores[class_idx] = []
                
                class_spectra[class_idx].append(X[i])
                class_attention_scores[class_idx].append(batch_attention_scores[i])
    
    # Compute mean spectra and attention scores per class
    for class_idx in sorted(class_spectra.keys()):
        spectra = np.array(class_spectra[class_idx])  # (n_samples, n_time_steps)
        attention_scores = np.array(class_attention_scores[class_idx])  # (n_samples, n_patches)
        
        mean_spectrum = spectra.mean(axis=0)  # (n_time_steps,)
        mean_attention = attention_scores.mean(axis=0)  # (n_patches,) - mean attention per patch
        
        # Plot single spectrum with patches colored by attention
        n_patches = len(mean_attention)
        
        fig, ax = plt.subplots(figsize=(14, 4))
        
        class_name = model.head.lb.classes_[class_idx]
        
        # Normalize attention scores to [0, 1] for coloring
        norm_attention = (mean_attention - mean_attention.min()) / (mean_attention.max() - mean_attention.min() + 1e-6)
        
        # Plot spectrum with patches colored by attention scores
        n_time_steps = len(mean_spectrum)
        patch_width = n_time_steps / n_patches
        
        cmap = plt.cm.Reds
        
        for patch_idx, att_score in enumerate(norm_attention):
            start = int(patch_idx * patch_width)
            end = int((patch_idx + 1) * patch_width)
            
            if start < n_time_steps and end <= n_time_steps:
                ax.fill_between(range(start, end), mean_spectrum[start:end], alpha=0.7, 
                               color=cmap(att_score))
        
        ax.plot(mean_spectrum, 'k-', linewidth=0.5, alpha=0.3)
        ax.set_xlabel('Time Steps', fontweight='bold')
        ax.set_ylabel('Amplitude', fontweight='bold')
        ax.grid(True, alpha=0.3)
        
        fig.suptitle(f'Mean Spectrum - Class: {class_name}\n(Patches colored by mean attention scores)', 
                     fontsize=14, fontweight='bold', y=0.995)
        
        # Add colorbar
        sm = plt.cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(vmin=mean_attention.min(), vmax=mean_attention.max()))
        sm.set_array([])
        cbar = plt.colorbar(sm, ax=ax, pad=0.02)
        cbar.set_label('Attention Score', fontweight='bold')
        
        plt.tight_layout()
        
        plt.savefig(os.path.join(run_dir, f'attention_spectrum_class_{class_name}.png'), 
                   dpi=300, bbox_inches='tight')
        plt.close()

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

    # Iterate over test data and collect predictions and labels
    with torch.no_grad():
        for batch in tqdm(test_loader, desc="Evaluating", leave=False):
            inputs = move_batch_to_device(batch, device)
            outputs = model(inputs)

            # For classification, we assume the model's head outputs logits for each class.
            # For regression, we assume the model's head outputs a single value.
            if task == "classification":
                targets = model.head.transform_labels(inputs['y_label'])
                targets = torch.argmax(targets, dim=1)
                preds = torch.argmax(outputs, dim=1)
            else:  # regression
                preds = outputs.squeeze()
                targets = inputs['y_label'].squeeze()
                
            all_preds.append(preds.detach().cpu())
            all_labels.append(targets.detach().cpu())
    
    all_preds = torch.cat(all_preds)
    all_labels = torch.cat(all_labels)

    metrics = {}
    if task == "classification":
        metrics["accuracy"] = accuracy_score(all_labels, all_preds)
        metrics["f1"] = f1_score(all_labels, all_preds, average="weighted")

        # Confusion matrix
        cm = confusion_matrix(all_labels, all_preds, normalize='true')
        disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=model.head.lb.classes_)
        
        fig, ax = plt.subplots(figsize=(12, 10))
        disp.plot(ax=ax, cmap='Reds', values_format='.1%')
        
        # Améliorer la lisibilité
        ax.set_xlabel('Prédiction', fontsize=12, fontweight='bold')
        ax.set_ylabel('Vérité', fontsize=12, fontweight='bold')
        ax.set_title('Matrice de Confusion', fontsize=14, fontweight='bold', pad=20)
        
        # Augmenter la taille des labels
        ax.xaxis.set_tick_params(labelsize=11)
        ax.yaxis.set_tick_params(labelsize=11)
        
        # Rotation des labels pour éviter le chevauchement
        plt.xticks(rotation=45, ha='right')
        plt.tight_layout()
        
        plt.savefig(os.path.join(run_dir, 'confusion_matrix.png'), dpi=300, bbox_inches='tight')
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

        # Save best model
        if valid_loss < best_valid_loss:
            best_valid_loss = valid_loss
            save_model_checkpoint(model, run_dir, name="best_model.pth")

        if scheduler:
            scheduler.step()

        print(f"Epoch {epoch}/{epochs} - Train Loss: {train_loss:.4f}, Valid Loss: {valid_loss:.4f}")

    # Plot training curves
    if run_dir:
        plot_metrics(all_train_losses, all_valid_losses, run_dir)
    
    # Save last model
    save_model_checkpoint(model, run_dir, name="last_model.pth")
    
    # Plot attention spectra for each class
    plot_attention_spectra(run_dir, model, test_loader, device, task=args.task)
    
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

    # Add mask ratio if the backbone is MAE
    # if args.backbone_init == "mae":
    #     results_dict["mask_ratio"] = args.mask_ratio

    results_dict.update(metrics)

    # Sauvegarde de la configuration du modèle
    save_model_config(run_dir, args)

    # Sauvegarde dans CSV
    log_metrics(results_dict)
    print(f"Results added to csv : {results_dict}")