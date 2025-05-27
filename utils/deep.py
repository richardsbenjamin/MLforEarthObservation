
import numpy as np

from _typing_ import ndarray


def get_device()


def normalise(image: ndarray) -> ndarray:
    return (image - np.mean(image)) / np.std(image)

def denormalise(image: ndarray, mean: float, std: float) -> ndarray:
    return (image + mean) / std



