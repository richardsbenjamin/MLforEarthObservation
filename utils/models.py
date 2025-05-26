from __future__ import annotations

import statsmodels.api as sm
from numpy import column_stack, ones_like
from scipy.stats import linregress

from _typing_ import TYPE_CHECKING

if TYPE_CHECKING:
    from _typing_ import ndarray, RegressionResultsWrapper


def linear_fit(index: ndarray, temp: ndarray) -> tuple[float]:
    if isinstance(index, (list, tuple)):
        index = index[0]
    res = linregress(index.flatten(), temp.flatten())
    return res.slope, res.intercept

def linear_sharpening(inputs: list[ndarray], linear_fit_res: tuple[float], *_) -> ndarray:
    return linear_fit_res[1] + inputs[0] * linear_fit_res[0]

def multi_linear_fit(
        inputs: list[ndarray],
        temp: ndarray,
    ) -> RegressionResultsWrapper:
    X = column_stack([input.flatten() for input in inputs])
    y = temp.flatten()
    X = sm.add_constant(X)
    return sm.OLS(y, X).fit()

def multi_linear_sharpening(
        inputs: list[ndarray],
        mutli_fit_res: RegressionResultsWrapper,
        reshape_size: int | None = None,
    ) -> ndarray:
    X = column_stack([input.flatten() for input in inputs])
    X = sm.add_constant(X)
    res = mutli_fit_res.predict(X)
    if reshape_size is None:
        return res
    return res.reshape((reshape_size, reshape_size))

def albedo_polynomial(index: ndarray, albedo: ndarray) -> ndarray:
    index = index.flatten()
    albedo = albedo.flatten()
    return column_stack([
        index**4, index**3 * albedo, index**2 * albedo**2, index * albedo**3,
        albedo**4, index**3, index**2 * albedo, index * albedo**2, albedo**3,
        index**2, index * albedo, albedo**2, index, albedo, ones_like(index)
    ])

def albedo_polynomial_fit(
        inputs: list[ndarray],
        temp: ndarray,
    ) -> RegressionResultsWrapper:
    x = albedo_polynomial(*inputs)
    return sm.OLS(temp.flatten(), x).fit()

def albedo_polynomial_sharpening(
        inputs: list[ndarray],
        poly_fit_res: RegressionResultsWrapper,
        reshape_size: int | None = None,
    ) -> ndarray:
    X = albedo_polynomial(*inputs)
    res = poly_fit_res.predict(X)
    if reshape_size is None:
        return res
    return res.reshape((reshape_size,reshape_size))
