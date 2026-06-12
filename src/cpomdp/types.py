from dataclasses import dataclass

import numpy as np
from numpy.typing import ArrayLike


@dataclass(frozen=True,  init=False)
class Belief:
    mean: np.ndarray
    cov: np.ndarray # Covariance

    @property
    def ndim(self) -> int:
        """Dimensionality of the state — the length of the mean vector."""
        return self.mean.shape[0]

    def __init__(self, mean: ArrayLike, cov: ArrayLike) -> None:
        object.__setattr__(self, "mean", np.asarray(mean, dtype=float))
        object.__setattr__(self, "cov", np.asarray(cov, dtype=float))
        self._validate()

    def _validate(self)-> None:
        if self.mean.ndim != 1:
            raise ValueError(
                f"belief mean must be a 1-D vector, got shape {self.mean.shape}"
            )
        if self.cov.ndim != 2:
            raise ValueError(
                f"belief covariance  must be a 2-D matrix, got shape {self.cov.shape}"
            )
        n = self.mean.shape[0]
        if self.cov.shape != (n, n):
            raise ValueError(
                f"belief covariance must be {n}x{n} to match a {n}-D mean, "
                f"got shape {self.cov.shape}"
            )
        if not np.allclose(self.cov, self.cov.T):
            raise ValueError(
                "belief covariance must be symmetrical."
            )

