import os
import numpy as np
from scipy.stats import norm, gamma, t
import matplotlib.pyplot as plt
import pyreadr
from utils.dist import compute_gaussian_scores, gaussian_pdf, student_pdf
from utils import madn
from utils.estimators import Huber, Bisquare, TrimmedMean

dataset_folder = 'rbtm_datasets'
def figure_3_1():
    total_samples = 50 + 1
    n_xs = 50
    data = np.asarray(norm.rvs(loc=0, scale=1, size=total_samples - 1), dtype=np.float32)
    std = np.std(data)
    madn_ = madn(data)

    x0_s = np.linspace(-5, 5, n_xs)
    # estimators
    huber = Huber(k=1.37)
    bisquare = Bisquare(k=4.68)
    trimmed = TrimmedMean(alpha=0.1)

    # initial estimators
    median_0 = np.median(data)
    huber_std_0 = huber.fit(data, dispersion=std)[0]
    huber_madn_0 = huber.fit(data, dispersion=madn_)[0]
    bisq_0 = bisquare.fit(data, dispersion=madn_)[0]
    trimmed_0 = trimmed.fit(data)[0]

    # arrays
    median_sc = np.zeros((n_xs,))
    huber_std = np.zeros((n_xs,))
    huber_madn = np.zeros((n_xs,))
    bisq_madn = np.zeros((n_xs,))
    trimmed_sc = np.zeros((n_xs,))

    for idex, x0 in enumerate(x0_s):
        data_ = np.append(data, x0)
        median_sc[idex] = total_samples * (np.median(data_) - median_0)
        huber_std[idex] = total_samples * (huber.fit(data_, dispersion=std)[0] - huber_std_0)
        huber_madn[idex] = total_samples * (huber.fit(data_, dispersion=madn_)[0] - huber_madn_0)
        bisq_madn[idex] = total_samples * (bisquare.fit(data_, dispersion=madn_)[0] - bisq_0)
        trimmed_sc[idex] = total_samples * (trimmed.fit(data_)[0] - trimmed_0)

    fig, axs = plt.subplots(2, 2)
    axs[0, 0].plot(x0_s, median_sc, lw=1)
    axs[0, 0].set_xlabel('x0')
    axs[0, 0].set_ylabel('SC')
    axs[0, 0].axhline(y=0, linestyle='--', color='k', lw=0.5)
    axs[0, 0].set_ylim(-2, 2)
    axs[0, 0].set_title('Median')
    axs[0, 1].plot(x0_s, huber_std, lw=1, color='b', label='std')
    axs[0, 1].plot(x0_s, huber_madn, lw=1, color='orange', label='madn')
    axs[0, 1].legend()
    axs[0, 1].set_xlabel('x0')
    axs[0, 1].set_ylabel('SC')
    axs[0, 1].axhline(y=0, linestyle='--', color='k', lw=0.5)
    axs[0, 1].set_ylim(-2, 2)
    axs[0, 1].set_title('Huber')
    axs[1, 0].plot(x0_s, bisq_madn, lw=1)
    axs[1, 0].set_xlabel('x0')
    axs[1, 0].set_ylabel('SC')
    axs[1, 0].axhline(y=0, linestyle='--', color='k', lw=0.5)
    axs[1, 0].set_ylim(-2, 2)
    axs[1, 0].set_title('Bisquare')
    axs[1, 1].plot(x0_s, trimmed_sc, lw=1)
    axs[1, 1].set_xlabel('x0')
    axs[1, 1].set_ylabel('SC')
    axs[1, 1].axhline(y=0, linestyle='--', color='k', lw=0.5)
    axs[1, 1].set_ylim(-2, 2)
    axs[1, 1].set_title('Trimmed')
    plt.show()

if __name__ == '__main__':
    figure_3_1()