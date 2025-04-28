import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from model import MAE1D
from torch.utils.data import DataLoader
from data.data.dataloader import CWRUDataset,custom_collate_fn

def mae_loss(predicted, original, mask):
    return ((predicted - original) ** 2 * mask).sum() / mask.sum()

size = 12048
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

# Define the MAE1D model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = MAE1D().to(device)

optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
criterion = nn.MSELoss()
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)

from tqdm import tqdm
num_epochs = 50
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

for epoch in range(num_epochs):
    model.train()
    epoch_loss = 0

    for batch in tqdm(train_loader):
        signals = torch.stack(batch['vibration']).unsqueeze(1).to(device)  # (batch_size, 1, signal_length)

        optimizer.zero_grad()
        predicted_signals, mask = model(signals)

        loss = mae_loss(predicted_signals, signals, mask)
        loss.backward()
        optimizer.step()

        epoch_loss += loss.item()

    print(f"Epoch {epoch}: loss = {epoch_loss / len(train_loader)}")
