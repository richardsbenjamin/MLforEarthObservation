from __future__ import annotations

from numpy import zeros

from _typing_ import TYPE_CHECKING

if TYPE_CHECKING:
    from _typing_ import Callable, ndarray


def get_target_by_classing(
        coarse_inputs: list[ndarray],
        fine_inputs: list[ndarray],
        coarse_target: ndarray,
        coarse_masks: dict,
        fine_masks: dict,
        fit_func: Callable,
        sharpening_func: Callable,
        *sharpening_args: tuple,
    ) -> tuple[ndarray]:
    n = len(coarse_inputs)
    if n != len(fine_inputs):
        raise ValueError
    
    n_h = fine_inputs[0].shape[0]
    n_c = coarse_inputs[0].shape[0]

    coarse_resids = zeros((n_c, n_c))
    fine_target = zeros((n_h, n_h))

    for class_ in coarse_masks.keys():
        coarse_mask = coarse_masks[class_]
        fine_mask = fine_masks[class_]

        coarse_masked_inputs = [input[coarse_mask] for input in coarse_inputs]
        fine_masked_inputs = [input[fine_mask] for input in fine_inputs]
        coarse_masked_target = coarse_target[coarse_mask]

        fit_res = fit_func(coarse_masked_inputs, coarse_masked_target)
        fit_pred = sharpening_func(coarse_masked_inputs, fit_res, *sharpening_args)
        coarse_resids[coarse_mask] = coarse_masked_target - fit_pred

        fine_target[fine_mask] = sharpening_func(fine_masked_inputs, fit_res, *sharpening_args)

    return fine_target, coarse_resids