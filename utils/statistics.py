import torch
import torch.nn as nn
from torch import Tensor

def mse_loss(x: Tensor, y: Tensor) -> Tensor:
    """
    Compute the Mean Squared Error (MSE).
    Args :
        x (Tensor): true signal
        y (Tensor): predict signal

    Returns : 
        Tensor: Mean Squared Error (MSE).
    """
    b, n = y.shape
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
    return (signal - mean[:N]) / std[:N]