import numpy as np
from typing import Union, Tuple

class Estimator:

    def __init__(self, epsilon: float=0.001, n_iter: int=100, verbosity: bool=False):
        self.epsilon = epsilon
        self.n_iter = n_iter
        self.verbose = verbosity

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

    def _newton_raphson(self, x: np.ndarray, dispersion: float) -> float:
        mu_0 = np.median(x)
        converged = False
        delta = 100
        for iter in range(self.n_iter):
            x_ = (x - mu_0) / dispersion
            mu_k = mu_0 + np.sum(self.psi(x_)) / np.sum(self.psi_prime(x_))
            if self.verbose:
                print(f'Iteration {iter + 1}:\n-mu_0 {mu_0}\nmu_k {mu_k}')
            delta = mu_k - mu_0
            mu_0 = mu_k
            if np.abs(delta) < self.epsilon:
                converged = True
                if self.verbose:
                    print(f'Algorithm converged after {iter + 1} iterations')
                break
        if not converged and self.verbose:
            print(f'Algorithm didnt converge after {self.n_iter} iterations.\n Delta {delta}')
        return mu_0, converged

    def _fit(self, x: np.ndarray, dispersion: float=None) -> Tuple[float, float, bool]:
        x = np.asarray(x)
        fit_dispersion = False if dispersion else True
        mu_hat = None
        converged = False
        if not fit_dispersion:
            mu_hat, converged = self._newton_raphson(x, dispersion=dispersion)
        return mu_hat, dispersion, converged

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
    def __init__(self, k: float):
        self.k = k
        super().__init__()

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

    def fit(self, x: np.ndarray, dispersion: float=None) -> Tuple[float, float]:
        return self._fit(x, dispersion)

    def asymptotic_variance(self, x: np.ndarray, dispersion: float, mu: float=None):
        return self._asymptotic_variance(x, dispersion=dispersion, mu=mu)

class Bisquare(Estimator):
    def __init__(self, k: float):
        self.k = k
        super().__init__()

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

    def fit(self, x: np.ndarray, dispersion: float=None) -> Tuple[float, float]:
        return self._fit(x, dispersion)

    def asymptotic_variance(self, x: np.ndarray, dispersion: float, mu: float=None):
        return self._asymptotic_variance(x, dispersion=dispersion, mu=mu)

class TrimmedMean(Estimator):
    def __init__(self, alpha: float):
        if alpha < 0 or alpha > 0.5:
            raise ValueError('Alpha must be within 0..0.5')
        self.alpha = alpha

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