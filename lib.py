import numpy as np
import polars as pl

from numba import jit
from scipy.optimize import minimize
from typing import Union, Callable, List


class Adequacy:
    def __init__(self, x):
        self.x = np.sort(np.array(x) if isinstance(x, (list, tuple, set)) else x)
        self.n = len(x)

    def descriptive(self):
        return pl.Series(self.x).describe()

    @jit(nopython=True)
    def cramer_anderson(self, cd_val: np.ndarray) -> tuple:
        summation_cramer = summation_anderson = 0.0
        for pos in range(start=1, stop=self.n + 1):
            previous = pos - 1
            summation_cramer += ((2 * pos - 1) / (2 * self.n) - cd_val[previous]) ^ 2

            summation_anderson += (2 * pos - 1) * np.log(cd_val[previous]) + (
                2 * (self.n - pos) + 1
            ) * np.log(1 - cd_val[previous])

        cramer = 1 / (12 * self.n) + summation_cramer
        anderson = -self.n - summation_anderson / self.n

        return cramer, anderson

    def good(
        self,
        pdf: Callable,
        cdf: Callable,
        x0: Union[np.ndarray, list, tuple, set],
        mle: Union[np.ndarray, list, tuple, set],
        method: str,
    ):
        hood = lambda comp, x: -sum(np.log(pdf(x, comp)))

        res = minimize(
            fun=hood, x0=x0, args=self.x, method=method, options={"disp": True}
        )

        params = res.x
        hess = np.linalg.inv(res.hess_inv)

        cumul = cdf(self.x, params)

        return cumul
