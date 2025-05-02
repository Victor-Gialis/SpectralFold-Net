import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np

from tqdm import tqdm
from model import ViT
from torch.utils.data import DataLoader

from utils import signal_normalization, global_stats
from data.data.dataloader import CWRUDataset,custom_collate_fn

# Size of the input signal
# The input signal is a 1D time series of length 2^14 (16384 samples).
# The signal is divided into patches of size 1024 (2^10) samples, and the model processes these patches.
size = 2**14 # 16384
recovery = 0.5 # 95% overlap between patches
# The stride is set to 5% of the window size, which means that the model will process 95% of the signal in each patch.

# The window size is set to 1024 samples (2^10), which means that the model will process 1024 samples at a time.
# The stride is set to 5% of the window size, which means that the model will process 95% of the signal in each patch.
dataset = CWRUDataset(
    fault_filter=['normal', 'inner', 'outer'],
    window_size=size,
    stride=int(size * (1 - recovery)))

# Compute global mean and std for normalization
# This is done to normalize the input signals before feeding them into the model.
mean, std = global_stats(dataset)
print(f"Global mean: {mean}, Global std: {std}")

# Create the training and validation datasets
# The dataset is split into training and validation sets using a random split.
train_size = int(0.8 * len(dataset)) # 80% pour l'entraînement
valid_size = len(dataset) - train_size # 20% pour la validation
train_dataset, valid_dataset = torch.utils.data.random_split(dataset, [train_size, valid_size]) # 80% train, 20% valid

# Create the test dataset
# The test dataset is created with a different fault filter (only 'ball' fault).
test_dataset = CWRUDataset(
    fault_filter=['ball'],
    window_size=size,
    stride=int(size * (1 - recovery)))

# Create DataLoader for training, validation, and test datasets
# The DataLoader is used to load the data in batches during training and evaluation.
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, collate_fn=custom_collate_fn)
valid_loader = DataLoader(valid_dataset, batch_size=32, shuffle=False, collate_fn=custom_collate_fn)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False, collate_fn=custom_collate_fn)

# Define the Vision Transformer 1D model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = ViT(serie_len=size).to(device)

mean = mean.to(device)
std = std.to(device)

# Define the optimizer and loss function
# The AdamW optimizer is used for training the model, and the loss function is Mean Squared Error (MSE).
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)

num_epochs = 150
for epoch in range(num_epochs):
    # Training
    model.train()
    epoch_loss = 0

    for batch in tqdm(train_loader):
        signals_reduce = torch.stack(batch['vibration_fft_reduce']).unsqueeze(1).to(device)  # (batch_size, 1, signal_length)
        signals_complete = torch.stack(batch['vibration_fft_complete']).unsqueeze(1).to(device)  # (batch_size, 1, signal_length)

        # Normalize the signals
        signals_reduce = signal_normalization(signals_reduce, mean, std) # (batch_size, 1, signal_length)
        signals_complete = signal_normalization(signals_complete, mean, std) # (batch_size, 1, signal_length)

        # Move signals to device
        optimizer.zero_grad()
        predicted_signals = model(signals_reduce) # (batch_size, 1, signal_length)

        # Reshape predicted signals to match the shape of the original signals
        loss = torch.mean(torch.abs(predicted_signals - signals_complete)) 
        loss.backward()
        optimizer.step()

        epoch_loss += loss.item()

    print(f"Epoch {epoch}: loss = {epoch_loss / len(train_loader)}")

    # Validation
    model.eval()
    val_loss = 0
    # Validation loop
    # The validation loop is similar to the training loop, but we do not compute gradients.
    with torch.no_grad():
        for batch in valid_loader:
            signals_reduce = torch.stack(batch['vibration_fft_reduce']).unsqueeze(1).to(device)
            signals_complete = torch.stack(batch['vibration_fft_complete']).unsqueeze(1).to(device)

            # Normalize the signals
            signals_reduce = signal_normalization(signals_reduce, mean, std) # (batch_size, 1, signal_length)
            signals_complete = signal_normalization(signals_complete, mean, std) # (batch_size, 1, signal_length)

            predicted_signals = model(signals_reduce)
            loss = torch.mean(torch.abs(predicted_signals - signals_complete))
            val_loss += loss.item()

    val_loss /= len(valid_loader)
    print(f"Epoch {epoch}: train loss = {epoch_loss / len(train_loader):.4f} | val loss = {val_loss:.4f}")

#Save the model
torch.save(model.state_dict(), 'model.pth')

# Load the model for inference
model = ViT(serie_len=size).to(device)
model.load_state_dict(torch.load('model.pth'))
model.eval()

# Test the model on the test set
test_loss = 0
for batch in tqdm(test_loader):
    signals_reduce = torch.stack(batch['vibration_fft_reduce']).unsqueeze(1).to(device)  # (batch_size, 1, signal_length)
    signals_complete = torch.stack(batch['vibration_fft_complete']).unsqueeze(1).to(device)  # (batch_size, 1, signal_length)

    with torch.no_grad():
        predicted_signals = model(signals_reduce)  # (batch_size, 1, signal_length)

    loss = torch.mean(torch.abs(predicted_signals - signals_complete))
    test_loss += loss.item()