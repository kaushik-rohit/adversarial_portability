import pandas as pd
from tqdm import tqdm
import pickle
import shutil
import pathlib
import shared
import os

def cp_data():
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
        for source in shared.model_names:
            row = [image_file, source]
            for target in shared.model_names:
                df2 = df.loc[(df['image'] == image_file) & (df['target'] == target)]
                # print(df2)
                assert(len(df2) == 1)
                if image_file not in true_labels:
                    true_labels[image_file] = df2.iloc[0]['true_label']
                val = df2.iloc[0][source]
                row.append(val)
            row.append(true_labels[image_file])
            rows.append(row)
    
    res_df = pd.DataFrame(rows, columns=cols)
    res_df.to_csv('output.csv', index=False)


def predictions_to_distance(path, cost_matrix, out_name):
    df = pd.read_csv(path)

    rows = []

    with open(cost_matrix, 'rb') as f:
        cost_matrix = pickle.load(f)

    for index, row in df.iterrows():
        entry = [row['image'], row['source']]
        true_label = row['true_label']

        for model in shared.model_names:
            entry.append(cost_matrix[true_label][row[model]])
        rows.append(entry)

    cols = df.columns[0:-1]

    res_df = pd.DataFrame(rows, columns=cols)

    res_df.to_csv(out_name, index=False)


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
            row.append('{}, {}'.format(mean, std))
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


if __name__ == '__main__':
    format_predictions('./results/fixed_alpha_with_preds/FGSM_alpha=0.01_preds.csv')
