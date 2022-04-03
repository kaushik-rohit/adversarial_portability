import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import shared

cols = shared.model_names

attacks = ['PGD', 'FGSM', 'FGV']
alphas = [0.01, 0.1, 0.3]

def heatmaps():
    fig, axes = plt.subplots(ncols=3, nrows=3, sharex=True, sharey=True)
    cbar_ax = fig.add_axes([.91, .3, .03, .4])
    
    for i in range(len(attacks)):
        for j in range(len(alphas)):
            axes[i][j].set_title('attack={}, alpha={}'.format(attacks[i], alphas[j]))
            df = pd.read_csv('./results/fixed_alpha/{}_alpha={}.csv'.format(attacks[i], alphas[j]), index_col=0)
            sns.heatmap(df, ax=axes[i][j], cmap="PiYG", xticklabels=True, yticklabels=True, cbar_ax=None if i else cbar_ax)
    plt.show()


if __name__ == '__main__':
    heatmaps()

