from typing import Union
import numpy as np
from scipy.stats import norm, gamma
from scipy.special import gamma as gamma_fn


def compute_gaussian_scores(n, get_confidence:bool = False):
    quantiles = (np.arange(1, n + 1) - 0.5) / n
    z_scores = norm.ppf(quantiles)

    upper_bound = None
    lower_bound = None
    if get_confidence:
        se = np.sqrt(quantiles * (1 - quantiles) / n) / norm.pdf(z_scores)
        upper_bound = z_scores + se
        lower_bound = z_scores - se
    return z_scores, upper_bound, lower_bound

def get_means(X, y=None):
    x_means = np.mean(X, axis=0)
    y_mean = None
    y_demean = None
    if len(y) > 0:
        y_mean = np.mean(y, axis=0)
        y_demean = y - y_mean
    return X - x_means, x_means, y_demean, y_mean

def fit_gaussian_MLE(X, y):
    covariance = X.T @ X
    return np.linalg.inv(covariance) @ X.T @ y

def gaussian_pdf(x: Union[np.ndarray, float], sigma:float=1, mu:float=0):
    #f(x)=exp(-(x-mu)^2/(2sigma^2)
    const = 1./(sigma * np.sqrt(2 * np.pi))
    exponent = -(x - mu) ** 2 / (2 * sigma ** 2)
    return const * np.exp(exponent)

def student_pdf(x: Union[np.ndarray, float], nu:float=1):
    base_term = (1 + x ** 2 / nu)
    exponent = -(nu + 1) / 2
    constant = gamma_fn((nu + 1) / 2) / (gamma_fn(nu / 2 ) * np.sqrt(np.pi * nu))
    return constant * (base_term ** exponent)