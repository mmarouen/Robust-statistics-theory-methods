import os
import numpy as np
from scipy.stats import norm, gamma, t
import matplotlib.pyplot as plt
import pyreadr
from utils.dist import compute_gaussian_scores, gaussian_pdf, student_pdf
from utils import get_iqr
from utils.estimators import Huber, Bisquare

dataset_folder = 'rbtm_datasets'
def figure_2_1():
    result = pyreadr.read_r(os.path.join(dataset_folder, 'flour.RData'))
    data = sorted(result['flour']['V1'].tolist())
    n_samples = len(data)
    z_scores, _, _ = compute_gaussian_scores(n_samples)
    plt.scatter(z_scores, data, facecolors='none', edgecolors='k')
    plt.xlabel('Quantiles of standard normal')
    plt.ylabel('Flout')
    plt.title('Figure 2.1 Q-Q plot of the flour data')
    plt.show()

def figure_2_2():
    data = np.linspace(-4, 4, 100)
    eps = 0.1
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

def figure_2_3():
    data = np.linspace(-3, 3, 100)
    k = 1.4
    huber = Huber(k)
    rho_graph = huber.rho(data)
    max_rho = max(rho_graph)
    labels = [-3, -2, '-k', -1, 0, 1, '+k', 2, 3]
    locs = [-3, -2, -k, -1, 0, 1, k, 2, 3]
    fig, axs = plt.subplots(2, 1)
    axs[0].plot(data, rho_graph)
    axs[0].axvline(x=+k, ymax=huber.rho(+k) / max_rho, ymin=0, linestyle='--', color='k', lw=0.5)
    axs[0].axvline(x=-k, ymax=huber.rho(-k) / max_rho, ymin=0, linestyle='--', color='k', lw=0.5)
    axs[0].set_xticks(ticks=locs, labels=labels)
    axs[0].set_xlabel('x')
    axs[0].set_ylabel('Rho')

    psi_graph = huber.psi(data)
    axs[1].plot(data, psi_graph)
    axs[1].plot([k, k], [0, huber.psi(k)], linestyle='--', color='k', lw=0.5)
    axs[1].plot([-k, -k], [0, huber.psi(-k)], linestyle='--', color='k', lw=0.5)
    axs[1].axhline(y=0, color='k', lw=0.5)
    axs[1].set_xticks(ticks=range(-3, 4), labels=range(-3, 4))
    axs[1].set_xlabel('x')
    axs[1].set_ylabel('Psi')
    fig.suptitle('Figure 2.3 Huber Rho and Psi functions')
    plt.show()

def figure_2_4():
    data = np.linspace(-3, 3, 100)
    k = 1.4
    fig, axs = plt.subplots(2, 1)
    huber = Huber(k)
    bisquare = Bisquare(k)
    axs[0].plot(data, huber.weight(data), lw=1)
    axs[0].set_xticks(ticks=range(-3, 4), labels=range(-3, 4))
    axs[0].set_xlabel('x')
    axs[0].set_ylabel('weight(x)')

    axs[1].plot(data, bisquare.weight(data), lw=1)
    axs[1].axhline(y=0, color='k', lw=0.5)
    axs[1].set_xticks(ticks=range(-3, 4), labels=range(-3, 4))
    axs[1].set_xlabel('x')
    axs[1].set_ylabel('weight(x)')
    fig.suptitle('Figure 2.4 Huber and Bisquare weight functions')
    plt.show()

def figure_2_5():
    data_boundary = 6
    data = np.linspace(-data_boundary, data_boundary, 100)
    k = 4.2
    fig, axs = plt.subplots(2, 1)
    bisquare = Bisquare(k)
    axs[0].plot(data, bisquare.rho(data), lw=1)
    axs[0].set_xticks(ticks=range(-data_boundary, data_boundary + 1), labels=range(-data_boundary, data_boundary + 1))
    axs[0].set_xlabel('x')
    axs[0].set_ylabel('Rho(x)')

    axs[1].plot(data, bisquare.psi(data), lw=1)
    axs[1].axhline(y=0, color='k', lw=0.5)
    axs[1].set_xticks(ticks=range(-data_boundary, data_boundary + 1), labels=range(-data_boundary, data_boundary + 1))
    axs[1].set_xlabel('x')
    axs[1].set_ylabel('Psi(x)')
    fig.suptitle('Figure 2.5 Bisquare Rho and Psi functions')
    plt.show()

# figure 2.6
initial_data = [
    28, 26, 33, 24, 34, -44, 27, 16, 40, -2,
    29, 22, 24, 21, 25, 30, 23, 29, 31, 19,
]
outliers = 3 * [44]
data = np.asarray(initial_data + outliers, dtype=np.float32)
mu_values = np.linspace(-60, 50, 500)
bisquare = Bisquare(k=4.65)
sigma_hat = np.median(np.abs(data - np.median(data))) / 0.675
psi_averages = []
rho_averages = []
for mu_i in mu_values:
    input_data = (data - mu_i) / sigma_hat
    psi_averages.append(np.mean(bisquare.psi(input_data)))
    rho_averages.append(np.mean(bisquare.rho(input_data)))

fig, axs = plt.subplots(2, 1)
axs[0].plot(mu_values, psi_averages, lw=1)
axs[0].axhline(y=0, linestyle='--', color='k', lw=0.5)
axs[0].set_xlabel('μ')
axs[0].set_ylabel('Average(Ψ)')

axs[1].plot(mu_values, rho_averages, lw=1)
axs[1].set_xlabel('μ')
axs[1].set_ylabel('Average(ρ)')
fig.suptitle('Figure 2.6 Psi, Rho of (x-mu)/sigma_hat')
plt.show()


#if __name__ == '__main__':
    #figure_2_5()