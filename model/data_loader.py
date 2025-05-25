import rasterio
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision.transforms import Resize

# Configuração do device
cuda = torch.cuda.is_available()
device = torch.device("cuda" if cuda else "cpu")

def read_images(path: str):
    with rasterio.open(path) as dataset:
        data = dataset.read(1).astype(np.float32)
        width = dataset.width
        height = dataset.height
        crs = dataset.crs
        transform = dataset.transform
    
    return data, width, height, crs, transform

def normalize_data(input_arr, target_arr):
    input_mean = np.mean(input_arr)
    input_std = np.std(input_arr)
    target_mean = np.mean(target_arr)
    target_std = np.std(target_arr)
    
    input_norm = (input_arr - input_mean) / input_std
    target_norm = (target_arr - target_mean) / target_std
    
    return input_norm, target_norm, (input_mean, input_std), (target_mean, target_std)

class TrajDataSet(Dataset):
    def __init__(self, paired_images, transform=None):
        self.paired_images = paired_images
        self.transform = transform

    def __len__(self):
        return len(self.paired_images)

    def __getitem__(self, idx):
        sample = {
            'input': self.paired_images[idx, 0],
            'target': self.paired_images[idx, 1]
        }
        if self.transform:
            sample = self.transform(sample)
        return sample

class ToTensor:
    def __call__(self, sample):
        input_tensor = torch.FloatTensor(sample['input']).unsqueeze(0) 
        target_tensor = torch.FloatTensor(sample['target']).unsqueeze(0)
        if cuda:
            input_tensor = input_tensor.to(device)
            target_tensor = target_tensor.to(device)
        return {'input': input_tensor, 'target': target_tensor}

def prepare_dataloader(input_arr, target_arr, batch_size=128, shuffle=True, img_size=224):
    input_norm, target_norm, input_stats, target_stats = normalize_data(input_arr, target_arr)
    
    if input_norm.shape != (img_size, img_size):
        resize = Resize((img_size, img_size))
        input_norm = resize(torch.from_numpy(input_norm).unsqueeze(0)).squeeze(0).numpy()
        target_norm = resize(torch.from_numpy(target_norm).unsqueeze(0)).squeeze(0).numpy()
    
    paired_images = np.stack((input_norm, target_norm), axis=0)  
    paired_images = np.expand_dims(paired_images, axis=0)  
    
    dataset = TrajDataSet(paired_images, transform=ToTensor())
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)
    
    return dataloader, input_stats, target_stats