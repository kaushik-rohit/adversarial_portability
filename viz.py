import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib as mpl
import matplotlib.cm as cm
import shared
import os

cols = shared.model_names

attacks = ['PGD', 'FGSM', 'FGV']
alphas = [0.01, 0.1, 0.3]

plt.rc('font', size=8)
plt.rc('xtick', labelsize=4)
plt.rc('ytick', labelsize=4)
plt.rc('axes', labelsize=4)
plt.rc('axes', titlesize=4)

def untargeted_heatmaps():
    '''
    generated 3*3 heatmaps for untargeted attacks corresponding
    to 3 attacks and 3 different perturbation constant
    '''
    fig, axes = plt.subplots(ncols=3, nrows=3, sharex=True, sharey=True)
    # cbar_ax = fig.add_axes([.91, .3, .03, .4])
    
    for i in range(len(attacks)):
        for j in range(len(alphas)):
            axes[i][j].set_title('attack={}, alpha={}'.format(attacks[i], alphas[j]))
            df = pd.read_csv('./results/fixed_alpha_updated/{}_alpha={}.csv'.format(attacks[i], alphas[j]), index_col=0)
            sns.heatmap(df, ax=axes[i][j], cmap="jet", xticklabels=True, yticklabels=True, vmin=0, vmax=10000)
    plt.tight_layout()
    plt.savefig("untargeted_heatmap.pdf", format="pdf")


def targeted_heatmaps():
    '''
    generated 3*3 heatmaps for targeted attacks corresponding
    to 3 attacks and 3 different perturbation constant
    '''
    targets = ['closest', 'farthest', 'median']

    min_max = {
        'closest' : [0, 10000],
        'farthest' : [0, 25],
        'median' : [0, 800]
    }

    for target in targets:
        fig, axes = plt.subplots(ncols=3, nrows=3, sharex=True, sharey=True)
        # cbar_ax = fig.add_axes([.91, .3, .03, .4])
        
        for i in range(len(attacks)):
            for j in range(len(alphas)):
                axes[i][j].set_title('attack={}, alpha={}'.format(attacks[i], alphas[j]))
                path = './results/target_attacks/{}/{}/{}'.format(target, attacks[i], alphas[j])
                filename = os.path.join(path, 'count_adversarial.csv')
                df = pd.read_csv(filename, index_col=0)
                sns.heatmap(df, ax=axes[i][j], cmap="jet", xticklabels=True, yticklabels=True, vmin=0, vmax=10000)
                # sns.heatmap(df, ax=axes[i][j], cmap="jet", xticklabels=True, yticklabels=True, vmin=min_max[target][0], vmax=min_max[target][1])
        plt.tight_layout()
        plt.savefig("targeted_{}_heatmap.pdf".format(target), format="pdf")
    

def line_plot():
    '''
    plot for perturbation constant vs portability
    '''
    models = ['resnet18', 'vgg16', 'googlenet', 'mobilenet_v3_small', 'densenet']

    fig, axes = plt.subplots(figsize=(7, 3), ncols=3, nrows=1, sharex=True, sharey=True)
    # cbar_ax = fig.add_axes([.91, .3, .03, .4])
    
    for i in range(len(attacks)):
        axes[i].set_title('attack={}'.format(attacks[i]))
        axes[i].set_xticks([0.01, 0.1, 0.3])
        df = pd.read_csv('./results/fixed_alpha/{}.csv'.format(attacks[i]))
        x = df['epsilon'].values
        for model in models:
            y = df[model].values
            axes[i].plot(x, y, label=model)
        axes[i].legend()
        axes[i].set_xlabel('epsilon')
        axes[i].set_ylabel('portability')
        # sns.heatmap(df, ax=axes[i][j], cmap="PiYG", xticklabels=True, yticklabels=True)
    plt.tight_layout()
    plt.savefig("line_plots.pdf", format="pdf")


def get_min_max(df):
    mean = []
    std = []
    for index, row in df.iterrows():
        for model in shared.model_names:
            rep = row[model].split(", ")
            mean.append(float(rep[0]))
            std.append(float(rep[1]))
    return min(mean), max(mean), min(std), max(std)

def plot_std_mean():
    '''
    plot std and mean of the distance between prediction and target class
    as circles for each of the selected neural networks.
    '''
    for attack in attacks:
        for alpha in alphas:
            fig, ax = plt.subplots(figsize=(7, 8))
            # plt.subplots_adjust(top=1.2)
            ax.axis('off')
            ax.axes.xaxis.set_visible(False)
            ax.axes.yaxis.set_visible(False)
            df = pd.read_csv('./results/fixed_alpha_with_preds/{}/{}/dis_metric.csv'.format(attack, alpha))
            min_mean, max_mean, min_std, max_std = get_min_max(df)
            print(min_mean, max_mean, min_std, max_std)
            norm = mpl.colors.Normalize(vmin=min_mean, vmax=max_mean)
            cmap = cm.jet
            m = cm.ScalarMappable(norm=norm, cmap=cmap)

            for ii in range(0, 13):
                row = df.iloc[ii]
                for jj in range(0, 13):
                    mean, std = row[shared.model_names[jj]].split(',')
                    mean = float(mean)
                    std = float(std)
                    std = (std - min_std) / (max_std - min_std)
                    std = std / 2
                    # print(mean, std)
                    circle = plt.Circle((ii, -1*jj), radius=std, color=m.to_rgba(mean))
                    ax.add_patch(circle)
                    ax.axis('equal')

            for ii in range(0, 13):       
                plt.annotate(xy= (ii, 1.5), text=shared.model_names[ii], fontsize = 8, verticalalignment='center', horizontalalignment='center', rotation=90)

            for jj in range(0, 13):
                plt.annotate(xy =(-2, -1*jj), text=shared.model_names[jj], fontsize = 8, verticalalignment='center', horizontalalignment='center')
            plt.tight_layout()
            # plt.colorbar()
            # plt.title('Mean and Std for distance between prediction on adversarial example and true label')
            plt.savefig("mean_plot_{}_{}.pdf".format(attack, alpha), format="pdf")

def plot_std_mean_targeted():
    targets = ['closest', 'farthest', 'median']
    for target in targets:
        for attack in attacks:
            for alpha in alphas:
                fig, ax = plt.subplots(figsize=(7, 8))
                # plt.subplots_adjust(top=1.2)
                ax.axis('off')
                ax.axes.xaxis.set_visible(False)
                ax.axes.yaxis.set_visible(False)
                df = pd.read_csv('./results/target_attacks/{}/{}/{}/dis_metric.csv'.format(target, attack, alpha))
                min_mean, max_mean, min_std, max_std = get_min_max(df)
                print(min_mean, max_mean, min_std, max_std)
                norm = mpl.colors.Normalize(vmin=min_mean, vmax=max_mean)
                cmap = cm.plasma
                m = cm.ScalarMappable(norm=norm, cmap=cmap)

                for ii in range(0, 13):
                    row = df.iloc[ii]
                    for jj in range(0, 13):
                        mean, std = row[shared.model_names[jj]].split(',')
                        mean = float(mean)
                        std = float(std)
                        std = (std - min_std) / (max_std - min_std)
                        std = std / 2
                        # print(mean, std)
                        circle = plt.Circle((ii, -1*jj), radius=std, color=m.to_rgba(mean))
                        ax.add_patch(circle)
                        ax.axis('equal')

                for ii in range(0, 13):       
                    plt.annotate(xy= (ii, 1.5), text=shared.model_names[ii], fontsize = 8, verticalalignment='center', horizontalalignment='center', rotation=90)

                for jj in range(0, 13):
                    plt.annotate(xy =(-2, -1*jj), text=shared.model_names[jj], fontsize = 8, verticalalignment='center', horizontalalignment='center')
                plt.tight_layout()
                # plt.colorbar()
                # plt.title('Mean and Std for distance between prediction on adversarial example and true label')
                plt.savefig("target_{}_mean_plot_{}_{}.pdf".format(target, attack, alpha), format="pdf")

if __name__ == '__main__':
    # targeted_heatmaps()
    # untargeted_heatmaps()
    plot_std_mean_targeted()
    # line_plot()
    # plot_std_mean()
