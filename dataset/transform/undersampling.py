import torch

def undersampling_time_serie(x, factor):
    """
    Reduce the sampling rate of a time series tensor by a given factor.
    Args:
        x (torch.Tensor): Input tensor representing the time series data.
        factor (int): Downsampling factor. Must be a positive integer.
    """
    if factor <= 0:
        raise ValueError("factor must be a positive integer.")
    
    return x[..., ::factor]
