import numpy as np
from numba import jit
import polars as pl

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
        summation_cramer = 0.0
        summation_anderson = 0.0
        for pos in range(start=1, stop=self.n + 1):
            summation_cramer += ((2 * pos - 1) / (2 * self.n) - cd_val[pos]) ^ 2

            summation_anderson += (2 * pos - 1) * np.log(cd_val[pos]) + (
                2 * (self.n - pos) + 1
            ) * np.log(1 - cd_val[pos])

        return (
            1 / (12 * self.n) + summation_cramer,
            -self.n - summation_anderson / self.n,
        )

    def good(self, pdf, cdf, starts, method, domain, mle, **kwargs):
        hood = lambda comp: -sum(pdf(comp, self.x))
        pass
