import os
from typing import Union
import numpy as np
from scipy.stats import norm, gamma, t
import matplotlib.pyplot as plt
import pandas as pd
import json
import pyreadr
from utils.dist import compute_gaussian_scores, gaussian_pdf, student_pdf

dataset_folder = 'rbtm_datasets'
# figure 2.1 Floor data
result = pyreadr.read_r(os.path.join(dataset_folder, 'flour.RData'))
data = sorted(result['flour']['V1'].tolist())
def figure_2_1():
    n_samples = len(data)
    z_scores, _, _ = compute_gaussian_scores(n_samples)
    plt.scatter(z_scores, data, facecolors='none', edgecolors='k')
    plt.xlabel('Quantiles of standard normal')
    plt.ylabel('Flout')
    plt.title('Figure 2.1 Q-Q plot of the flour data')
    plt.show()

# figure 2.2
def get_iqr(data):
    q75, q25 = np.percentile(data, [75 ,25])
    return q75 - q25
data = np.linspace(-4, 4, 100)
eps = 0.1
def figure_2_2():
    normal = gaussian_pdf(data)
    contaminated = (1 - eps) * gaussian_pdf(data) + eps * gaussian_pdf(data, sigma=100)
    iqr_t4 = t.ppf(0.75, df=4) - t.ppf(0.25, df=4)
    t4_scaled = student_pdf(data / iqr_t4, nu=4) / iqr_t4

    gaussian_iqr = get_iqr(normal)
    #t4_scaled = t4 * gaussian_iqr / get_iqr(t4)
    contaminated_scaled = contaminated * gaussian_iqr / get_iqr(contaminated)
    plt.plot(data, normal, color='k')
    #plt.plot(data, contaminated_scaled, color='b')
    plt.plot(data, t4_scaled, color='g')
    plt.xlabel('x')
    plt.ylabel('Densities')
    plt.title('Figure 2.2 Standard normal (N), Student (T4) contaminated normal (CN) densities scaled to interquantile ranges')
    plt.show()

# figure 2.3
data = np.linspace(-3, 3, 100)
k = 1.2
def rho_k(x: Union[float, np.array], k:float = 1.4):
    return np.where(np.abs(x) <= k, x ** 2, 2 * k * np.abs(x) - k ** 2)

rho_graph = rho_k(data, k)
plt.plot(data, rho_graph)
plt.show()