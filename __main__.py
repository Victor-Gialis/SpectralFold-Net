import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np

from tqdm import tqdm
from model import ViT
from torch.utils.data import DataLoader
from data.data.dataloader import CWRUDataset,custom_collate_fn

# Size of the input signal
# The input signal is a 1D time series of length 2^14 (16384 samples).
# The signal is divided into patches of size 1024 (2^10) samples, and the model processes these patches.
size = 2**14 # 16384
recovery = 0.95

train_dataset = CWRUDataset(
    fault_filter=['normal', 'inner', 'outer'],
    window_size=size,
    stride=int(size * (1 - recovery)))

test_dataset = CWRUDataset(
    fault_filter=['ball'],
    window_size=size,
    stride=int(size * (1 - recovery)))

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, collate_fn=custom_collate_fn)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False, collate_fn=custom_collate_fn)

# Define the Vision Transformer 1D model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = ViT(serie_len=size).to(device)

optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
criterion = nn.MSELoss()

num_epochs = 1
for epoch in range(num_epochs):
    model.train()
    epoch_loss = 0

    for batch in tqdm(train_loader):
        signals_reduce = torch.stack(batch['vibration_fft_reduce']).unsqueeze(1).to(device)  # (batch_size, 1, signal_length)
        signals_complete = torch.stack(batch['vibration_fft_complete']).unsqueeze(1).to(device)  # (batch_size, 1, signal_length)

        optimizer.zero_grad()
        predicted_signals = model(signals_reduce) # (batch_size, 1, signal_length)

        # Reshape predicted signals to match the shape of the original signals
        loss = torch.mean(torch.abs(predicted_signals - signals_complete))
        loss.backward()
        optimizer.step()

        epoch_loss += loss.item()

    print(f"Epoch {epoch}: loss = {epoch_loss / len(train_loader)}")
