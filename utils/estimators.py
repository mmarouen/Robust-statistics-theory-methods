import numpy as np
from typing import Union

class Estimator:
    def rho(self, x: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
        raise NotImplementedError

    def psi(self, x: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
        raise NotImplementedError

    def weight(self, x: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
        raise NotImplementedError

class Huber(Estimator):
    def __init__(self, k: float):
        self.k = k

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

    def weight(self, x: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
        scalar_input = np.isscalar(x)
        x = np.asarray(x)
        result = np.where(x == 0, 1, np.minimum(1, self.k / np.abs(x)))
        return float(result) if scalar_input else result

class Bisquare(Estimator):
    def __init__(self, k: float):
        self.k = k

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

    def weight(self, x: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
        scalar_input = np.isscalar(x)
        x = np.asarray(x)
        result = np.where(np.abs(x) <= self.k, ((1 - (x / self.k) ** 2) ** 2), 0)
        return float(result) if scalar_input else result
