import torch
from model.data_loader import (
    read_images,
    prepare_dataloader
)
from training.train import train_model

path_index_h = "data/NDBI_20m.img"
path_temperature_c = "data/LST_100m.img"
path_index_c = "data/NDBI_100m.img"

I_H, cols_h, rows_h, crs_h, transform_h = read_images(path_index_h)
T_C, cols_t, rows_t, crs_t, transform_t = read_images(path_temperature_c)
I_C, cols_c, rows_c, crs_c, transform_c = read_images(path_index_c)

model, input_stats, target_stats = train_model(I_C, T_C, epochs=50, batch_size=32)

torch.save(model.state_dict(), "training/swin_transformer_model.pth")