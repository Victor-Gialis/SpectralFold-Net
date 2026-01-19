import torch

def reduced_magnitude_spectrum(x):
    """
    Compute the reduced magnitude spectrum of a time series tensor.
    Args:
        x (torch.Tensor): numpy array or of length N representing the time series data.
    Returns:
        torch.Tensor: Tensor containing the reduced magnitude spectrum.
    """
    # psd = torch.abs(torch.fft.rfft(data - torch.mean(data)))/N

    x -= torch.mean(x, dim=-1, keepdim=True)  # Center the signal
    n = len(x)

    # Compute the Fast Fourier Transform (FFT)
    fft_result = torch.fft.rfft(x)
    
    # Compute the magnitude spectrum
    reduced_spectrum = torch.abs(fft_result)/n

    return reduced_spectrum[:-1] # Exclude the Nyquist frequency component

def symetric_spectrum_preparation(spectrum, factor=1):
    """
    Prepare a symmetric spectrum by concatenating the lipped version of the spectrum.
    Args:
        spectrum (torch.Tensor): Input tensor representing the spectrum data.
        factor (int): Number of times to concatenate the flipped spectrum.
    Returns:
        torch.Tensor: Tensor containing the symmetric spectrum.

    """
    n = len(spectrum)
    complete_spectrum = list()

    for i in range(1,factor):
        if i//2 == 0 and factor > 1:
            complete_spectrum.append(torch.flip(spectrum, [-1]))
        else:
            complete_spectrum.append(spectrum)
    
    complete_spectrum = torch.cat((complete_spectrum), dim=-1)
    spectrum = torch.cat((spectrum, complete_spectrum), dim=-1)
    return spectrum