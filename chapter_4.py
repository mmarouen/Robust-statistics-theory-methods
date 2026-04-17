import os
import numpy as np
import matplotlib.pyplot as plt
import pyreadr
from pydantic import BaseModel
from rstm.estimators import Bisquare, Category, L1, L2

dataset_folder = 'rbtm_datasets'
def figure_4_1_table_4_1():

    result = pyreadr.read_r(os.path.join(dataset_folder, 'shock.RData'))
    data = result['shock']
    x = np.asarray(data['n.shocks'], dtype=np.float32)
    y = np.asarray(data['time'], dtype=np.float32)
    line_x = np.linspace(0, 15, 50)

    # first line fit
    X = np.hstack((np.ones([x.shape[0], 1], x.dtype), np.expand_dims(x, -1)))
    # 2nd line fit (without points 0, 1, 3)
    to_remove = [0, 1, 3]
    x_ = np.delete(x, to_remove)
    y_ = np.delete(y, to_remove)
    X_ = np.hstack((np.ones([x_.shape[0], 1], x.dtype), np.expand_dims(x_, -1)))

    bisq = Bisquare(k=4., category=Category.REGRESSION.value, verbosity=True)
    beta_bisq, _, _ = bisq.fit(x=X, y=y, dispersion=np.std(y))

    l2 = L2(category=Category.REGRESSION.value, verbosity=True)
    beta_l2_all, _, _ = l2.fit(x=X, y=y, dispersion=np.std(y))
    beta_l2_clean, _, _ = l2.fit(x=X_, y=y_, dispersion=np.std(y))

    l1 = L1(category=Category.REGRESSION.value, verbosity=True)
    beta_l1, _, _ = l1.fit(x=X_, y=y_, dispersion=np.std(y_))
    print(f'Beta bisq {beta_bisq}\nBeta LS (cleaned) {beta_l2_clean}\nBeta L1 {beta_l1}')
    fig, axs = plt.subplots(1, 2)

    axs[0].scatter(x, y, facecolors='none', edgecolors='k')
    axs[0].plot(line_x, beta_l2_all[1] * line_x + beta_l2_all[0], color='k', lw=0.5, label='regular')
    axs[0].scatter(x[to_remove], y[to_remove], c='red', label='omitted points') 
    axs[0].plot(line_x, beta_l2_clean[1] * line_x + beta_l2_clean[0], color='cyan', lw=0.5, label='omitting points')
    axs[0].set_xlabel('number of shocks')
    axs[0].set_ylabel('average response time')
    axs[0].set_title('Fig 4.1 Shock data: LS fit with all data and omitting points 1,2 and 4')
    axs[0].legend(loc='lower center', ncol=3)

    class TableRow(BaseModel):
        '''For plotting the table'''
        Model: str
        Intercept: float
        Slope: float

    bisquare_row = TableRow(Model='Bisquare', Intercept=beta_bisq[0], Slope=beta_bisq[1])
    l1_row = TableRow(Model='L1', Intercept=beta_l1[0], Slope=beta_l1[1])
    l2_all = TableRow(Model='LS', Intercept=beta_l2_all[0], Slope=beta_l2_all[1])
    l2_clean = TableRow(Model='LS (-1, 2,4)', Intercept=beta_l2_clean[0], Slope=beta_l2_clean[1])

    colnames = list(bisquare_row.model_dump().keys())
    values = [
        list(l2_all.model_dump().values()),
        list(l2_clean.model_dump().values()),
        list(l1_row.model_dump().values()),
        list(bisquare_row.model_dump().values()),
        ]

    axs[1].axis("off")
    axs[1].set_title('Table 4.1 Regression estimates for rats data')
    table = axs[1].table(
        colLabels=colnames,
        cellText=[[f"{v:.2f}" if isinstance(v, float) else str(v) for v in row]
            for row in values],
        loc="center"
        )
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1.2, 1.2)
    plt.show()




if __name__ == '__main__':
    figure_4_1_table_4_1()
