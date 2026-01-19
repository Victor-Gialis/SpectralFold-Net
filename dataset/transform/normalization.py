import torch

def min_max_log_normalization(x:torch.Tensor, stats_from:torch.Tensor, eps=1e-8):
    """
    Log + min-max normalization using stats_from for consistent scaling.
    Normalization is done per sample.
    """
    x_log = torch.log1p(x)
    stats_log = torch.log1p(stats_from)

    x_min = stats_log.amin(dim=-1, keepdim=True)
    x_max = stats_log.amax(dim=-1, keepdim=True)

    x_norm = (x_log - x_min) / (x_max - x_min + eps)

    return x_norm

def min_max_log_unnormalization(x_norm:torch.Tensor, stats_from:torch.Tensor, eps=1e-8):
    stats_log = torch.log1p(stats_from)

    x_min = stats_log.amin(dim=-1, keepdim=True)
    x_max = stats_log.amax(dim=-1, keepdim=True)

    x_log = x_norm * (x_max - x_min + eps) + x_min
    x_unnorm = torch.expm1(x_log)
    
    return x_unnorm
