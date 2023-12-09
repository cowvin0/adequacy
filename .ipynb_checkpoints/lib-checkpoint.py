import numpy as np
import polars as pl

from numba import jit
from functools import partial
from scipy.optimize import minimize
from typing import Union, Callable, List


class Adequacy:
    def __init__(self, x: Union[np.ndarray, list]):
        self.x = np.array(x) if isinstance(x, list) else x
        self.n = len(x)

    def descriptive(self):
        return pl.Series(self.x).describe()

    @jit(nopython=True)
    def cramer_anderson(self, cd_val):
        summation_cramer = summation_anderson = 0.0
        for pos in range(start=1, stop=self.n + 1):
            previous = pos - 1
            summation_cramer += ((2 * pos - 1) / (2 * self.n) - cd_val[previous]) ^ 2

            summation_anderson += (2 * pos - 1) * np.log(cd_val[previous]) + (
                2 * (self.n - pos) + 1
            ) * np.log(1 - cd_val[previous])

        return (
            1 / (12 * self.n) + summation_cramer,
            -self.n - summation_anderson / self.n,
        )

    def good(
        self, pdf, x0, method
    ):  # cdf=None, x0, method, domain=None, mle=None, **kwargs):
        hood = lambda comp, x: -sum(pdf(x, comp))

        res = minimize(
            fun=hood, x0=[1], args=(self.x,), method=method, options={"xatol": 1e-8}
        )

        return res
