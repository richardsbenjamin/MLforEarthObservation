from __future__ import annotations

import rasterio
from numpy import float32, mean

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

