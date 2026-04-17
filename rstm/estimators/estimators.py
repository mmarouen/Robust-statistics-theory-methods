import numpy as np
from typing import Union, Tuple
from enum import Enum

class Category(Enum):
    REGRESSION = 1
    LOCATION = 2

class Solver(Enum):
    NEWTON_RAPHSON = 1
    ITERATIVE_REWEIGHTED_LS = 2


class Estimator:

    def __init__(self, category: int, epsilon: float=1e-4, n_iter: int=100, verbosity: bool=False):
        self.epsilon = epsilon
        self.n_iter = n_iter
        self.verbose = verbosity
        self.category = category

    def rho(self, x: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
        raise NotImplementedError

    def psi(self, x: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
        raise NotImplementedError

    def psi_prime(self, x: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
        raise NotImplementedError

    def weight(self, x: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
        raise NotImplementedError

    def fit(self, x: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
        raise NotImplementedError

    def asymptotic_variance(self, x: np.ndarray, dispersion: float, mu: float=None):
        raise NotImplementedError

    def _solve(
            self,
            x: np.ndarray,
            dispersion: float,
            y: np.ndarray = None,
            solver: int = Solver.ITERATIVE_REWEIGHTED_LS.value
        ) -> float:
        mu_0 = np.median(x)
        beta_0 = np.median(x, axis=0)

        converged = False
        delta = 100
        if self.verbose:
            print(f'Fitting {self.__class__.__name__}')
        for iter in range(self.n_iter):
            # init residuals
            if self.category == Category.LOCATION.value:
                residuals = (x - mu_0) / dispersion
            else:
                residuals = (y - x @ beta_0)
                dispersion = 1.4826 * np.median(np.abs(residuals))
                residuals /= dispersion

            # update step
            if solver == Solver.ITERATIVE_REWEIGHTED_LS.value:
                w = self.weight(residuals)
                if self.category == Category.LOCATION.value:
                    mu_k = np.sum(x * w) / np.sum(w)
                else:
                    Xw = x * w[:, None]
                    beta_k = np.linalg.solve(Xw.T @ x, Xw.T @ y)
            else:
                mu_k = mu_0 + np.sum(self.psi(residuals)) / np.sum(self.psi_prime(residuals))
            if self.category == Category.LOCATION.value:
                delta = np.abs(mu_k - mu_0) / dispersion
            elif self.category == Category.REGRESSION.value:
                delta = np.max(np.abs(beta_0 - beta_k)) / (np.max(np.abs(beta_0)) + 1e-8)

            # log progress
            if self.verbose and self.category == Category.LOCATION.value:
                print(f'Iteration {iter + 1}:\nmu_0 {mu_0}\nmu_k {mu_k}\nDelta {delta}')
            elif self.verbose and self.category == Category.REGRESSION.value:
                print(f'Iteration {iter + 1}: Delta {delta:.4f}')

            # update params
            if self.category == Category.LOCATION.value:
                mu_0 = mu_k
            elif self.category == Category.REGRESSION.value:
                beta_0 = beta_k

            if delta < self.epsilon:
                converged = True
                if self.verbose:
                    print(f'Algorithm converged after {iter + 1} iterations')
                break
        if not converged and self.verbose:
            print(f'Algorithm didnt converge after {self.n_iter} iterations.\n Delta {delta}')
        param = mu_0
        if self.category == Category.REGRESSION.value:
            param = beta_0
        return param, converged

    def _fit(self, x: np.ndarray, dispersion: float=None, y: np.ndarray = None) -> Tuple[Union[float, np.ndarray], float, bool]:
        x = np.asarray(x)
        fit_dispersion = False if dispersion else True
        if self.category == Category.REGRESSION.value and y is None:
            raise('Regression model could only be fit with a valid target variable')
        if self.category == Category.REGRESSION.value:
            solver = Solver.ITERATIVE_REWEIGHTED_LS.value
        else:
            solver = Solver.NEWTON_RAPHSON.value

        fitted_param = None
        converged = False
        if not fit_dispersion:
            fitted_param, converged = self._solve(x, dispersion=dispersion, y=y, solver=solver)
        return fitted_param, dispersion, converged

    def _asymptotic_variance(self, x: np.ndarray, dispersion: float, mu: float=None):
        x = np.asarray(x)
        if mu is None:
            mu, _, _ = self._fit(x, dispersion)
        input_data = (x - mu) / dispersion
        return (
                dispersion ** 2 *
                np.mean(self.psi(input_data) ** 2) /
                (np.mean(self.psi_prime(input_data)) ** 2)
            )

class Huber(Estimator):
    def __init__(self, k: float, **kwrargs):
        self.k = k
        super().__init__(**kwrargs)

    def rho(self, x: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
        scalar_input = np.isscalar(x)
        x = np.asarray(x)
        result = np.where(np.abs(x) <= self.k, x ** 2, 2 * self.k * np.abs(x) - self.k ** 2)
        return float(result) if scalar_input else result

    def psi(self, x: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
        scalar_input = np.isscalar(x)
        x = np.asarray(x)
        result = np.where(np.abs(x) <= self.k, x, self.k * np.sign(x))
        return float(result) if scalar_input else result

    def psi_prime(self, x: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
        scalar_input = np.isscalar(x)
        x = np.asarray(x)
        result = np.where(np.abs(x) <= self.k, 1., 0.)
        return float(result) if scalar_input else result

    def weight(self, x: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
        scalar_input = np.isscalar(x)
        x = np.asarray(x)
        result = np.where(x == 0, 1, np.minimum(1, self.k / np.abs(x)))
        return float(result) if scalar_input else result

    def fit(self, x: np.ndarray, dispersion: float=None, y: np.ndarray = None) -> Tuple[float, float]:
        return self._fit(x, dispersion)

    def asymptotic_variance(self, x: np.ndarray, dispersion: float, mu: float=None):
        return self._asymptotic_variance(x, dispersion=dispersion, mu=mu)

class Bisquare(Estimator):
    def __init__(self, k: float, **kwrargs):
        self.k = k
        super().__init__(**kwrargs)

    def rho(self, x: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
        scalar_input = np.isscalar(x)
        x = np.asarray(x)
        result = np.where(np.abs(x) <= self.k, 1 - ((1 - (x / self.k) ** 2) ** 3), 1)
        return float(result) if scalar_input else result

    def psi(self, x: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
        scalar_input = np.isscalar(x)
        x = np.asarray(x)
        result = np.where(np.abs(x) <= self.k, x * ((1 - (x / self.k) ** 2) ** 2), 0)
        return float(result) if scalar_input else result

    def psi_prime(self, x: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
        scalar_input = np.isscalar(x)
        x = np.asarray(x)
        result = np.where(np.abs(x) <= self.k, (1 - (x / self.k) ** 2) * (1 - 5 * (x / self.k) ** 2), 0.)
        return float(result) if scalar_input else result

    def weight(self, x: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
        scalar_input = np.isscalar(x)
        x = np.asarray(x)
        result = np.where(np.abs(x) <= self.k, ((1 - (x / self.k) ** 2) ** 2), 0)
        return float(result) if scalar_input else result

    def fit(self, x: np.ndarray, dispersion: float=None, y: np.ndarray = None) -> Tuple[float, float, bool]:
        return self._fit(x, dispersion, y=y)

    def asymptotic_variance(self, x: np.ndarray, dispersion: float, mu: float=None):
        return self._asymptotic_variance(x, dispersion=dispersion, mu=mu)

class L1(Estimator):
    def __init__(self, **kwrargs):
        super().__init__(**kwrargs)

    def rho(self, x: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
        scalar_input = np.isscalar(x)
        x = np.asarray(x)
        result = np.abs(x)
        return float(result) if scalar_input else result

    def psi(self, x: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
        scalar_input = np.isscalar(x)
        x = np.asarray(x)
        result = np.sign(x)
        return float(result) if scalar_input else result

    def psi_prime(self, x: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
        scalar_input = np.isscalar(x)
        x = np.asarray(x)
        result = np.zeros_like(x)
        return float(result) if scalar_input else result

    def weight(self, x: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
        scalar_input = np.isscalar(x)
        x = np.asarray(x)
        result = 1./np.maximum(np.abs(x), 1e-6)
        return float(result) if scalar_input else result

    def fit(self, x: np.ndarray, dispersion: float=None, y: np.ndarray = None) -> Tuple[float, float, bool]:
        return self._fit(x, dispersion, y=y)

    def asymptotic_variance(self, x: np.ndarray, dispersion: float, mu: float=None):
        return self._asymptotic_variance(x, dispersion=dispersion, mu=mu)

class L2(Estimator):
    def __init__(self, **kwrargs):
        super().__init__(**kwrargs)

    def rho(self, x: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
        scalar_input = np.isscalar(x)
        x = np.asarray(x)
        result = x ** 2
        return float(result) if scalar_input else result

    def psi(self, x: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
        scalar_input = np.isscalar(x)
        x = np.asarray(x)
        result = x
        return float(result) if scalar_input else result

    def psi_prime(self, x: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
        scalar_input = np.isscalar(x)
        x = np.asarray(x)
        result = np.ones_like(x)
        return float(result) if scalar_input else result

    def weight(self, x: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
        scalar_input = np.isscalar(x)
        x = np.asarray(x)
        result = np.ones_like(x)
        return float(result) if scalar_input else result

    def fit(self, x: np.ndarray, dispersion: float=None, y: np.ndarray = None) -> Tuple[float, float, bool]:
        return self._fit(x, dispersion, y=y)

    def asymptotic_variance(self, x: np.ndarray, dispersion: float, mu: float=None):
        return self._asymptotic_variance(x, dispersion=dispersion, mu=mu)

class TrimmedMean(Estimator):
    def __init__(self, alpha: float, **kwrargs):
        if alpha < 0 or alpha > 0.5:
            raise ValueError('Alpha must be within 0..0.5')
        self.alpha = alpha
        super().__init__(**kwrargs)

    def fit(self, x: np.ndarray) -> Union[float, np.ndarray]:
        sorted_data = np.asarray(sorted(x))
        n_sample = len(x)
        m = int(np.floor(self.alpha * n_sample))
        return np.mean(sorted_data[m: -m]), None, None

    def asymptotic_variance(self, x, mu = None):
        sorted_data = np.asarray(sorted(x))
        n_sample = len(x)
        m = int(np.floor(self.alpha * n_sample))
        if mu is None:
            mu, _, _ = self.fit(sorted_data)
        return (
            np.sum((sorted_data[m: -m] - mu) ** 2) +
            m * (sorted_data[m - 1] - mu) ** 2 +
            m * (sorted_data[-m] - mu) ** 2
        ) / (n_sample - 2 * m)