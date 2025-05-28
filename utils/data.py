from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
import rasterio
from numpy import array, float32, mean

from _typing_ import TYPE_CHECKING

if TYPE_CHECKING:
    from _typing_ import ndarray

class FilePaths:

    INDEX_C  = "data/NDBI_100m.img"
    INDEX_H  = "data/NDBI_20m.img"
    TEMP_C   = "data/LST_100m.img"
    TEMP_H   = "data/LST_20m.img"
    ALBEDO_C = "data/Albedo_100m.img"
    ALBEDO_H = "data/Albedo_20m.img"
    CLASS_C  = "data/Class_100m.img"
    CLASS_H  = "data/Class_20m.img"


def read_image_file(file_path: str) -> ndarray:
    with rasterio.open(file_path) as dataset:
        return dataset.read(1).astype(float32)
    
def rmse(x: ndarray, y: ndarray) -> float:
    return mean((x - y) ** 2)

def mean_bias_error(true_values, predicted_values):
    true_values = array(true_values)
    predicted_values = array(predicted_values)
    return mean(true_values - predicted_values)

def correlation_coefficient(true_values, predicted_values):
    true_values = array(true_values)
    predicted_values = array(predicted_values)
    numerator = np.sum(true_values * predicted_values)
    denominator = np.sqrt(np.sum(true_values**2) * np.sum(predicted_values**2))
    return numerator / denominator


def display_image(image: ndarray) -> None:
    fig, ax0 = plt.subplots(nrows=1, ncols=1, sharex=False, figsize=(18, 9))

    ax0.set_title('Fine resolution LST')
    img=ax0.imshow(image, cmap='inferno')

    fig.subplots_adjust(right=0.8)
    cbar_ax = fig.add_axes([0.85, 0.15, 0.02, 0.7])
    cbar=fig.colorbar(img, cax=cbar_ax)

    plt.show()

