from __future__ import annotations

from numpy import array, float32, zeros

from _typing_ import TYPE_CHECKING

if TYPE_CHECKING:
    from _typing_ import Callable, ndarray


def extract_patches(image: ndarray, patch_size: int, stride: int) -> ndarray:
    h, w = image.shape
    if h != w:
        raise ValueError

    patches = []
    for y in range(0, h - patch_size + 1, stride):
        for x in range(0, w - patch_size + 1, stride):
            patch = image[y:y + patch_size, x:x + patch_size]
            patches.append(patch)

    return array(patches)

def get_all_patches(inputs: list[ndarray], patch_size: int, stride: int) -> list[ndarray]:
    inputs_patches = []
    for input in inputs:
        inputs_patches.append(
            extract_patches(input, patch_size, stride)
        )
    return inputs_patches

def get_fine_reconstruction_from_patches(
        coarse_inputs: list[ndarray],
        fine_inputs: list[ndarray],
        coarse_target: list[ndarray],
        fit_func: Callable,
        sharpening_func: Callable,
        patch_size: int,
        stride: int,
        *sharpening_args: tuple
    ) -> ndarray:
    n = len(coarse_inputs)
    if n != len(fine_inputs):
        raise ValueError
    
    n_h = fine_inputs[0].shape[0]
    n_c = coarse_inputs[0].shape[0]
    scale = int(n_h / n_c)

    coarse_inputs_patches = get_all_patches(coarse_inputs, patch_size, stride)
    fine_inputs_patches = get_all_patches(fine_inputs, patch_size*scale, stride*scale)
    coarse_temp_patches = extract_patches(coarse_target, patch_size, stride)

    fine_temp_patches = []
    coarse_residual_patches = []

    for tuple_ in zip(*coarse_inputs_patches, *fine_inputs_patches, coarse_temp_patches):
        coarse_patch_inputs = tuple_[:n]
        fine_patch_inputs = tuple_[n:2*n]
        temp_patch = tuple_[-1]

        fit_res = fit_func(coarse_patch_inputs, temp_patch)
        fit_pred = sharpening_func(coarse_patch_inputs, fit_res, patch_size, *sharpening_args)
        coarse_residual_patches.append(temp_patch - fit_pred)

        fine_temp_patch = sharpening_func(fine_patch_inputs, fit_res, patch_size*scale, *sharpening_args)
        fine_temp_patches.append(fine_temp_patch)

    return (
        reconstruct_image_from_patches(
            array(fine_temp_patches), n_h, patch_size*scale, stride*scale,
        ),
        reconstruct_image_from_patches(
            array(coarse_residual_patches), n_c, patch_size, stride,
        ), 
    )


def reconstruct_image_from_patches(
        patches: list[ndarray],
        image_size: int,
        patch_size: int,
        stride: int,
    ) -> ndarray:
    reconstructed = zeros((image_size, image_size), dtype=float32)
    weight = zeros((image_size, image_size), dtype=float32)

    patch_idx = 0
    for y in range(0, image_size - patch_size + 1, stride):
        for x in range(0, image_size - patch_size + 1, stride):
            reconstructed[y:y+patch_size, x:x+patch_size] += patches[patch_idx]
            weight[y:y+patch_size, x:x+patch_size] += 1
            patch_idx += 1

    weight[weight == 0] = 1
    return reconstructed / weight

