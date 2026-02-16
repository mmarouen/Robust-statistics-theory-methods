import os
import numpy as np
from scipy.stats import norm, gamma
import matplotlib.pyplot as plt
import pandas as pd
import json
from utils.dist import compute_gaussian_scores, get_means, fit_gaussian_MLE

dataset_folder = 'rbtm_datasets'

# figure 1.2: 3 sigma rule
data = np.asarray([
    28, 26, 33, 24, 34, -44,  27, 16, 40,
    -2, 29, 22, 24, 21, 25, 30, 23, 29, 31, 19
    ], dtype=np.float16) # copied from book


def figure_1_2(dataset):
    n = len(dataset)
    dataset.sort()
    x_hat = np.mean(dataset)
    std = np.std(dataset)
    scores = (dataset - x_hat) / std
    theoretical_scores, _, _ = compute_gaussian_scores(n)
    print(f'Theoretical normal quantiles: {theoretical_scores}\nEmpirical quantiles: {scores}')
    fig, ax1 = plt.subplots()
    ax2 = ax1.twinx()
    ax1.plot(theoretical_scores, scores, "bo", mfc='none')
    ax1.set_xlabel('Quantiles of the Standard Normal')
    ax1.set_ylabel('Standardized empirical scores')
    ax2.plot(theoretical_scores, dataset, "go", mfc='none')
    ax2.set_ylabel('Raw input data')
    fig.suptitle('Figure 1.2 Velocity of light:Q-Q plot of observed times')
    plt.show()

# figure 1.4 1.5 (not including robust statistics)

data = pd.read_csv(os.path.join(dataset_folder, 'wagnerGrowth.csv'))
data.drop(['rownames'], axis=1, inplace=True)
regions = pd.get_dummies(data['Region'], drop_first=True, prefix='region') * 1
periods = pd.get_dummies(data['Period'], drop_first=True, prefix='period') * 1
data.drop(['Region', 'Period'], axis=1, inplace=True)
data = pd.concat([data, regions, periods], axis=1)
y = data['y'].to_numpy()
inputs = data.drop(['y'], axis=1).to_numpy()

def figures_1_4_and_1_5(inputs, y):
    n_samples = len(y)
    inputs_demean, x_means, y_demean, y_mean = get_means(inputs, y)
    X = np.column_stack([np.ones(inputs_demean.shape[0]), inputs_demean])
    print(f'Input predictors {X.shape}, target variable {y_demean.shape}')
    beta = fit_gaussian_MLE(X, y_demean)

    y_hat = np.matmul(X, beta)
    residuals = y_demean - y_hat
    sigma_hat = np.std(residuals, ddof=1)
    normalized_residuals = residuals / sigma_hat
    fig, axs = plt.subplots(1, 2, sharey=True)
    fig.subplots_adjust(wspace=0)

    ylim = 6
    axs[0].plot(range(n_samples), normalized_residuals)
    axs[0].axhline(y=2.33, linestyle='--')
    axs[0].axhline(y=-2.33, linestyle='--')
    axs[0].set_ylim(-ylim, ylim)
    axs[0].set_xlabel('Index (Time)')
    axs[0].set_ylabel('Standardized residuals')
    axs[0].set_title('Residuals over time plot')
    sorted_normalized = sorted(normalized_residuals)
    zscores, upper_bound, lower_bound = compute_gaussian_scores(n_samples, get_confidence=True)
    axs[1].scatter(zscores, sorted_normalized)
    axs[1].scatter(zscores, upper_bound, s=5, c='black')
    axs[1].scatter(zscores, lower_bound, s=5, c='black')
    axs[1].plot([np.min(zscores), np.max(zscores)], [np.min(zscores), np.max(zscores)], 'k-')
    axs[1].set_ylim(-ylim, ylim)
    axs[1].set_xlabel('Normal z-scores')
    axs[1].set_title('Residuals normal Q-Q plot')
    fig.suptitle('Figures 1.4-1.5 For regular linear regression')
    plt.show()
