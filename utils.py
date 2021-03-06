import pandas as pd
from tqdm import tqdm
import pickle
import shutil
import pathlib
import shared
import numpy as np
import os

cost_matrix_path = './data/cost_matrix.pkl'

def cp_data():
    '''
    copies the selected data 10 image per 1000 class which 
    are classified correctly by all of our neural network to a new location
    '''
    with open('dataset.pkl', 'rb') as f:
        data = pickle.load(f)

    classes = data.keys()

    for _class in classes:
        paths = data[_class]
        for path in paths:
            src = path[0]
            class_id = src.split('/')[-2]
            file_name = src.split('/')[-1]
            trg = '/local/scratch/rohit/datasets/ImageNet/ILSVRC2012/train/{}'.format(class_id)
            pathlib.Path(trg).mkdir(parents=True, exist_ok=True)
            trg = os.path.join(trg, file_name)
            shutil.copyfile(src, trg)

def format_predictions(filename):
    # print(filename)
    df = pd.read_csv(filename)
    image_files = list(df['image'].unique())
    rows = []
    cols = ['image', 'source'] + shared.model_names + ['true_label']
    true_labels = {}

    for image_file in tqdm(image_files):
        df2 = df.loc[df.image == image_file]

        assert(len(df2) == len(shared.model_names))
        assert (list(df2['target'].values) == shared.model_names)

        true_labels[image_file] = df2.iloc[0]['true_label']

        for model in shared.model_names:
            row = [image_file, model]
            val = df2[model].T
            row.extend(val)
            row.append(true_labels[image_file])
            rows.append(row)
    
    res_df = pd.DataFrame(rows, columns=cols)
    res_df.to_csv(filename, index=False)


def predictions_to_distance(path, cost_matrix, out_name, target='true_label'):
    '''
    @path: location to predictions
    @cost_matix: path to matrix containing cost of missclassification
    @out_name: output path
    @target: true_label in case of untargeted attack and target in case of targeted attack
    '''
    df = pd.read_csv(path)

    rows = []

    with open(cost_matrix, 'rb') as f:
        cost_matrix = pickle.load(f)

    for index, row in df.iterrows():
        entry = [row['image'], row['source']]
        true_label = row[target]

        for model in shared.model_names:
            entry.append(cost_matrix[true_label][row[model]])
        rows.append(entry)

    if target == 'target':
        cols = df.columns[0:-2]
    else:
        cols = df.columns[0:-1]

    res_df = pd.DataFrame(rows, columns=cols)

    res_df.to_csv(out_name, index=False)

def helper(attack, alpha):
    path = './results/fixed_alpha_with_preds/{}/{}'.format(attack, alpha)
    filename = os.path.join(path, '{}_alpha={}_preds.csv'.format(attack, alpha))
    cost_matrix = './data/cost_matrix.pkl'
    out_name = os.path.join(path, 'dis.csv')
    predictions_to_distance(filename, cost_matrix, out_name)
    get_predictions_distance_metric(out_name, os.path.join(path, 'dis_metric.csv'))

def helper2(attack, alpha, target):
    path = './results/target_attacks/{}/{}/{}'.format(target, attack, alpha)
    filename = os.path.join(path, 'targeted_{}_alpha={}_preds.csv'.format(attack, alpha))
    cost_matrix = './data/cost_matrix.pkl'
    out_name = os.path.join(path, 'dis.csv')

    format_predictions(filename)
    get_target(filename, target)
    predictions_to_distance(filename, cost_matrix, out_name, target='target')
    get_predictions_distance_metric(out_name, os.path.join(path, 'dis_metric.csv'))


def get_target(output_path, target_option):
    with open(cost_matrix_path, 'rb') as f:
        cost_matrix = pickle.load(f)

    def get_target_label(x, target_option='closest'):
        values = np.array(list(cost_matrix[x].values()))

        if target_option == 'closest':
            return np.where(values == np.min(values[np.nonzero(values)]))[0][0]
        elif target_option == 'farthest':
            return np.where(values == np.max(values[np.nonzero(values)]))[0][0]
        elif target_option == 'median':
            return np.where(values == np.median(values[np.nonzero(values)]))[0][0]

    df = pd.read_csv(output_path)

    df['target'] = df['true_label'].apply(lambda x: get_target_label(x, target_option))

    df.to_csv(output_path, index=False)

def get_predictions_distance_metric(path, outpath):
    df = pd.read_csv(path)

    rows = []

    cols = ['source'] + shared.model_names

    for source in shared.model_names:
        df_source = df[df.source == source]
        assert(len(df_source) == 10000)
        row = [source]

        for target in shared.model_names:
            mean = df_source[target].mean()
            std = df_source[target].std()
            row.append('{:.2f}, {:.2f}'.format(mean, std))
        rows.append(row)
    res_df = pd.DataFrame(rows, columns=cols)

    res_df.to_csv(outpath, index=False)


def get_cost_matrix():
    tree = shared.ImageNetHeirarchy('./data/imgnet_heirarchy', './data/wnid_to_label.csv')

    rows = []
    cols = ['class'] + list(range(1000))
    cost_matrix = {x: {y: -1 for y in range(1000)} for x in range(1000)}

    for i in tqdm(range(1000), total=1000):
        row = [i]
        for j in range(1000):
            if cost_matrix[i][j] != -1:
                row.append(cost_matrix[i][j])
            elif j == i:
                row.append(0)
                cost_matrix[i][j] = 0
            else:
                cost_matrix[i][j] = tree.get_lca(i, j)
                cost_matrix[j][i] = cost_matrix[i][j]
                row.append(cost_matrix[i][j])
        rows.append(row)
    res_df = pd.DataFrame(rows, columns=cols)

    res_df.to_csv('cost_matrix.csv', index=False)


def get_best_portability(attack, alpha):
    path = './results/fixed_alpha/{}_alpha={}.csv'.format(attack, alpha)
    df = pd.read_csv(path)

    rows = []
    for index, row in df.iterrows():
        max_port = 0
        min_port = 10001
        max_model = ''
        min_model = ''
        for model in shared.model_names:
            if model == row['target']:
                continue
            if max_port < row[model]:
                max_port = row[model]
                max_model = model
            if min_port > row[model]:
                min_port = row[model]
                min_model = model
        max_string = '{} ({})'.format(max_model, max_port)
        min_string = '{} ({})'.format(min_model, min_port)
        rows.append([row['target'], max_string, min_string])

    res_df = pd.DataFrame(rows, columns=['source', 'maximum', 'minimum'])
    res_df.to_csv('max_min_{}_{}.csv'.format(attack, alpha), index=False)


if __name__ == '__main__':
    # format_predictions('./results/fixed_alpha_with_preds/FGSM_alpha=0.01_preds.csv')
    get_best_portability('PGD', 0.3)
