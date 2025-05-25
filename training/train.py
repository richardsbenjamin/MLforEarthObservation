import torch
import torch.nn as nn
import torch.optim as optim
from model.swin_transformer import SwinTransformerV1
from model.data_loader import prepare_dataloader
import numpy as np

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def train_model(input_data, target_data, epochs=50, batch_size=32, learning_rate=1e-4):
    train_loader, input_stats, target_stats = prepare_dataloader(input_data, target_data, batch_size)
    
    # Cria o modelo simplificado
    model = SwinTransformerV1(
        img_size=224,
        in_chans=1,
        out_chans=1
    ).to(device)
    
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    
    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        
        for batch in train_loader:
            inputs = batch['input'].to(device)
            targets = batch['target'].to(device)
            
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
        
        print(f'Epoch {epoch+1}/{epochs}, Loss: {running_loss/len(train_loader):.4f}')
    
    return model, input_stats, target_stats