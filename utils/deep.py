import numpy as np

import torch
import torch.nn as nn
from torch.utils.data import Dataset

from _typing_ import ndarray


def stack_inputs(inputs: tuple, output: ndarray) -> tuple[ndarray]:
    x = np.stack(inputs, axis=1)
    y = output.flatten()[:, np.newaxis]
    return x, y

def train(model, criterion, optimizer, dataloader, dataset_length, checkpoint_name, n_epochs=50) -> None:
    best_loss = float('inf')
    for epoch in range(n_epochs):
        running_loss = 0.0
        for inputs, targets in dataloader:
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()
            running_loss += loss.item() * inputs.size(0)

        epoch_loss = running_loss / dataset_length
        
        if epoch_loss < best_loss:
            best_loss = epoch_loss
            torch.save(model.state_dict(), f"./checkpoints/{checkpoint_name}_epoch_{epoch+1}.pt")
            print(f"Epoch {epoch+1}/{n_epochs}, Loss: {epoch_loss:.4f}")

class RegressionDataset(Dataset):
    def __init__(self, X, Y, transform=None):
        self.X = X
        self.Y = Y
        self.transform = transform

    def __len__(self):
        return self.X.shape[0]

    def __getitem__(self, idx):
        x = self.X[idx]
        y = self.Y[idx]
        if self.transform:
            x = self.transform(x)
            y = self.transform(y)
        return x, y
    
class Regressor(nn.Module):
    def __init__(self, in_c):
        super().__init__()
        self.lin1 = nn.Sequential( 
            nn.Linear(in_c, 256, bias = False),
            nn.BatchNorm1d(256),
            nn.ReLU(True),
            )
        self.lin2 = nn.Sequential(
            nn.Linear(256, 512, bias = False),
            nn.BatchNorm1d(512),
            nn.ReLU(True),
            )
        self.lin3 = nn.Sequential(
            nn.Linear(512, 1, bias = False),
            nn.BatchNorm1d(1),
            )
        
    def forward(self, z):    
        out  = self.lin1(z)
        out  = self.lin2(out)
        out  = self.lin3(out)
        return out

class ToTensor(object):

    def __init__(self, device: str = "cuda:0") -> None:
        self.device = device

    def __call__(self, sample):
        return torch.FloatTensor(sample).to(self.device)