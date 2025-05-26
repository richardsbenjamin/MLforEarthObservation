from __future__ import annotations

from numpy import array, linspace, ones_like, zeros
from pykrige.ok import OrdinaryKriging

from _typing_ import TYPE_CHECKING

if TYPE_CHECKING:
    from _typing_ import ndarray


def upscale_coarse(
        coarse_image: ndarray, 
        scale: int = 5,
    ) -> ndarray:
    h, w = coarse_image.shape
    if h != w:
        raise ValueError

    upscaled_image = zeros((h * scale, h * scale))

    for i in range(h):
        for j in range(h):
            coarse_pixel = coarse_image[i][j]

            i_s = i * scale
            j_s = j * scale

            for i_h in range(scale):
                for j_h in range(scale):
                    upscaled_image[i_s+i_h][j_s+j_h] = coarse_pixel

    return upscaled_image

def upscale_coarse_krig(coarse_image: ndarray, scale: int = 5) -> ndarray:
    if coarse_image.shape[0] != coarse_image.shape[1]:
        raise ValueError
    n = coarse_image.shape[0]
    coarse_coords = []
    coarse_values = []

    for i in range(n):
        for j in range(n):
            coarse_coords.append((i, j))
            coarse_values.append(coarse_image[i, j])

    coarse_coords = array(coarse_coords)
    coarse_values = array(coarse_values)

    fine_x = linspace(0, n - 1, n * scale)
    fine_y = linspace(0, n - 1, n * scale)

    OK = OrdinaryKriging(
        x=coarse_coords[:, 1],  
        y=coarse_coords[:, 0],
        z=coarse_values,
        variogram_model='exponential',
        verbose=False,
        enable_plotting=False
    )

    fine_image, _ = OK.execute('grid', fine_x, fine_y)
    return array(fine_image)

