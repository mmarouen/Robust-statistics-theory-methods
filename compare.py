import numpy as np
import matplotlib.pyplot as plt
import scipy.stats
from typing import List, Tuple

""" N = 1_000
M = 10_000

v_mads: List[float] = []
v_stds: List[float] =[]
for _ in range(M):
    gauassian_samples = np.random.normal(loc=0, scale=1, size=N)
    pareto_samples =   """

# fig 3.26 spurious PCA
def get_eigenvalues(p: int, n: int, dist: str) -> Tuple[np.ndarray, np.ndarray]:
    X: np.ndarray = None
    if dist == 'normal':
        X = scipy.stats.norm.rvs(loc=0., scale=1., size=n * p).reshape((n, p))
    else:
        X = scipy.stats.levy_stable.rvs(alpha=1.5, beta=1, loc=0., scale=1., size=n * p).reshape((n, p))
    cov = (X.T @ X) / (n - 1)
    eigen_values, _ = np.linalg.eigh(cov)
    eigen_vals = -np.sort(-eigen_values)
    return eigen_vals * 100 / np.sum(eigen_vals), X, cov

p = 30
n1 = 1000
n2 = 1_000_000
fig, axs = plt.subplots(2, 1)
eigen_1, g1, _ = get_eigenvalues(p, n1, dist='normal')
eigen_2, g2, corr = get_eigenvalues(p, n2, dist='normal')
axs[0].bar(range(1, p + 1), eigen_1, color='orange', label=f'{n1:,}')
axs[0].bar(range(1, p + 1), eigen_2, fill=False, label=f'{n2:,}')
axs[0].set_title(f'Gaussian distribution')
axs[0].legend(title='Sample size')
axs[0].set_xticks([])
axs[0].set_ylabel('Percent of total information')
print(f'Covariance matrix for gaussian {corr[0,:]}')

eigen_1, a1, _ = get_eigenvalues(p, n1, dist='alpha')
eigen_2, a2, corr = get_eigenvalues(p, n1, dist='alpha')
axs[1].bar(range(1, p + 1), eigen_1, color='orange', label=f'{n1:,}')
axs[1].bar(range(1, p + 1), eigen_2, fill=False, label=f'{n2:,}')
axs[1].set_title(f'Long tailed (volatile) distribution')
axs[1].legend(title='Sample size')
idx = [1, 10, 20, 30]
axs[1].set_xticks(idx)
axs[1].set_xticklabels(idx)
axs[1].set_xlabel('Factors')
axs[1].set_ylabel('Percent of total information')
print(f'Covariance matrix for alpha stable {corr[0,:]}')

plt.suptitle(f'Patterns appearing in random data generated from {p} factors', fontweight='bold')
plt.show()

fig, axs = plt.subplots(2, 1)
axs[0].hist(g2[:, 0])
X = scipy.stats.levy_stable.rvs(alpha=1.5, beta=1, loc=0., scale=1., size=n2 * p).reshape((n2, p))
v1 = X[:, 0]
p1, p99 = np.percentile(v1, [1, 90])
print(min(v1), max(v1))
axs[1].hist(v1, bins=100)
plt.show()