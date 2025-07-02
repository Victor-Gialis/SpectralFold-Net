import torch
import torch.nn as nn
from torch import Tensor

import torch
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
    return torch.mean((x - y)**2)

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
    for sample in dataset:
        signal = sample['X_true']
        all_signals.append(signal)

    stacked = torch.stack(all_signals)
    mean = stacked.mean(axis=0)
    std = stacked.std(axis=0)
    return mean, std

def signal_normalization(signal: Tensor, mean: Tensor, std: Tensor) -> Tensor:
    """
    Normalize the input signal using the provided mean and standard deviation.
    
    Args:
        signal (Tensor): The input signal to be normalized.
        mean (Tensor): The mean value for normalization.
        std (Tensor): The standard deviation value for normalization.

    Returns:
        Tensor: The normalized signal.
    """
    N = signal.shape[-1]
    return (signal - mean[:,:,:N]) / std[:,:,:N]