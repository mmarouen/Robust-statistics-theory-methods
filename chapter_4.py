import os
import numpy as np
import matplotlib.pyplot as plt
import pyreadr
from typing import Tuple

dataset_folder = 'rbtm_datasets'
def figure_4_1():
    def rss_fit(x: np.ndarray, y: np.ndarray) -> Tuple[float, float]:
        n_samples = len(x)
        y_mean = np.mean(y)
        x_mean = np.mean(x)
        beta_1 = (
            (np.sum(x * y) - n_samples * x_mean * y_mean) /
            (np.sum(x ** 2) - n_samples * (x_mean ** 2)))
        beta_0 = y_mean - beta_1 * x_mean
        return beta_0, beta_1

    result = pyreadr.read_r(os.path.join(dataset_folder, 'shock.RData'))
    data = result['shock']
    x = np.asarray(data['n.shocks'], dtype=np.float32)
    y = np.asarray(data['time'], dtype=np.float32)
    line_x = np.linspace(0, 15, 50)

    # first line fit
    beta_0, beta_1 = rss_fit(x, y)
    # 2nd line fit (without points 0, 1, 3)
    to_remove = [0, 1, 3]
    x_ = np.delete(x, to_remove)
    y_ = np.delete(y, to_remove)
    beta_0_, beta_1_ = rss_fit(x_, y_)

    plt.scatter(x, y, facecolors='none', edgecolors='k')
    plt.plot(line_x, beta_1 * line_x + beta_0, color='k', lw=0.5)
    plt.plot(line_x, beta_1_ * line_x + beta_0_, color='cyan', lw=0.5)
    plt.xlabel('number of shocks')
    plt.ylabel('average response time')
    plt.show()

if __name__ == '__main__':
    figure_4_1()
