import torch
import torch.nn as nn
from torch import Tensor

import torch
from tqdm import tqdm
from torch import Tensor

def custom_loss(x: Tensor, y: Tensor, alpha: float = 1.0, beta: float = 0.5, ceta: float = 0.5) -> Tensor:
    """
    Compute a custom loss combining MSE, Pearson correlation, and Cosine similarity.
    Loss = alpha * MSE + beta * (1 - Pearson) + ceta * (1 - Cosine similarity)

    Args:
        x (Tensor): True signal (batch_size, signal_length).
        y (Tensor): Predicted signal (batch_size, signal_length).
        alpha (float): Weight for MSE loss.
        beta (float): Weight for Pearson correlation loss.
        ceta (float): Weight for Cosine similarity loss.

    Returns:
        Tensor: Combined loss value.
    """
    # Ensure x and y have the same length
    b, n = y.shape
    x = x[:, :n]

    # Compute MSE loss
    mse = torch.mean((x - y) ** 2)

    # Compute Pearson correlation
    x_mean = torch.mean(x, dim=-1, keepdim=True)
    y_mean = torch.mean(y, dim=-1, keepdim=True)
    x_centered = x - x_mean
    y_centered = y - y_mean
    numerator = torch.sum(x_centered * y_centered, dim=-1)
    denominator = torch.sqrt(torch.sum(x_centered ** 2, dim=-1) * torch.sum(y_centered ** 2, dim=-1))
    pearson = numerator / (denominator + 1e-8)  # Add epsilon to avoid division by zero
    pearson_loss = 1 - pearson.mean()

    # Compute Cosine similarity
    cosine_similarity = torch.nn.functional.cosine_similarity(x, y, dim=-1)
    cosine_loss = 1 - cosine_similarity.mean()

    # Combine losses
    total_loss = alpha * mse + beta * pearson_loss + ceta * cosine_loss
    return mse, pearson_loss, cosine_loss, total_loss

def mse_loss(x: Tensor, y: Tensor) -> Tensor:
    """
    Compute the Mean Squared Error (MSE).
    Args :
        x (Tensor): true signal
        y (Tensor): predict signal

    Returns : 
        Tensor: Mean Squared Error (MSE).
    """
    b,c, n = y.shape
    x = x[:,:n]
    # return torch.mean((x - y)**2)/torch.mean(x**2)
    return torch.mean((x - y)**2)


def regularized_loss(x: Tensor, y: Tensor, model, lambda_l1 = 0.0, lambda_l2 = 0.0):
    loss = mse_loss(x, y)
    l1_penalty = 0.0
    l2_penalty = 0.0

    for name, param in model.parameters():
        if "biais" not in name :
            if lambda_l1 > 0:
                l1_penalty += torch.sum(torch.abs(param))
            if lambda_l2 > 0:
                l2_penalty += torch.sum(param ** 2)

    total_loss = loss + lambda_l1 * l1_penalty + lambda_l2 * l2_penalty
    return total_loss

def covariance_matrix(x: Tensor) -> Tensor:
    x -= torch.mean(x, dim=-1, keepdim=True)
    cov = (x.T @ x).T / (x.shape[-1] - 1)
    return cov

def covariance_loss(x: Tensor, y: Tensor) -> Tensor:
    """
    Compute the covariance matrix of two signals.

    Args :
        x (Tensor): true signal
        y (Tensor): predict signal

    Returns :
        Tensor: covariance matrix of the two signals
    """ 
    b, n = y.shape
    x = x[:,:n]
    Cov_true = covariance_matrix(x)
    Cov_pred = covariance_matrix(y)
    return torch.norm(Cov_true - Cov_pred)**2

def global_stats(dataset):
    """
    Compute the global mean and standard deviation of the dataset.
    """
    all_signals = []
    for sample in tqdm(dataset, desc="Computing global stats"):
        signal = sample['X_true']
        all_signals.append(signal)

    stacked = torch.stack(all_signals)
    mean = stacked.mean(axis=0)
    std = stacked.std(axis=0)
    return mean, std

def _z_norm(x: Tensor, mean: Tensor, std: Tensor) -> Tensor:
    """
    Normalize the input signal using the provided mean and standard deviation.
    
    Args:
        signal (Tensor): The input signal to be normalized.
        mean (Tensor): The mean value for normalization.
        std (Tensor): The standard deviation value for normalization.

    Returns:
        Tensor: The normalized signal.
    """
    N = x.shape[-1]
    return (x - mean[:,:,:N]) / std[:,:,:N]

def _log_norm(x : Tensor) -> Tensor :
    """
    Normalise chaque spectre individuellement
    x: tensor de shape [batch_size,channel,lenght]
    """
    x_log = torch.log1p(x)
    x_max = x_log.max(dim=-1, keepdim=True).values
    x_min = x_log.min(dim=-1, keepdim=True).values
    x_norm = (x_log - x_min) / (x_max - x_min + 1e-8)
    return x_norm

def _log_denorm(x ,x_norm : Tensor)-> Tensor :
    """
    Dénormalise chaque spectre individuellement
    x: tensor de shape [batch_size,channel,lenght]
    """
    x_max = x.max(dim=-1, keepdim=True).values
    x_min = x.min(dim=-1, keepdim=True).values
    x_denorm = x_norm * (x_max - x_min + 1e-8) + x_min
    x_exp = torch.expm1(x_denorm)
    return x_exp