import pandas as pd
import os


model_names = [
    'resnet18',
    'resnet34',
    'resnet50',
    'resnet101',
    'resnet152',
    'vgg16',
    'vgg19',
    'googlenet',
    'densenet',
    'mobilenet_v2',
    'mobilenet_v3_small',
    'mobilenet_v3_large',
    'wide_resnet_50_2'
]

def  count(attack, alpha, target):
    path = './results/target_attacks/{}/{}/{}'.format(target, attack,  alpha)
    filename = os.path.join(path, 'targeted_{}_alpha={}_preds.csv'.format(attack, alpha))
    df = pd.read_csv(filename)
    df = df.drop('image', axis=1)
    for model in model_names:
        df[model] = df.apply(lambda x: 1 if x[model] == x['target'] else 0, axis=1)
    df = df.drop(['true_label', 'target'], axis=1)

    res_df = df.groupby(['source'], sort=False).sum()

    res_df.to_csv(os.path.join(path, 'count_target.csv'))

    df = pd.read_csv(filename)
    df = df.drop('image', axis=1)
    for model in model_names:
        df[model] = df.apply(lambda x: 1 if x[model] != x['true_label'] else 0, axis=1)
    df = df.drop(['true_label', 'target'], axis=1)

    res_df = df.groupby(['source'], sort=False).sum()

    res_df.to_csv(os.path.join(path, 'count_adversarial.csv'))


if __name__ == '__main__':
    attacks = ['PGD', 'FGSM',  'FGV']
    targets = ['closest',  'farthest', 'median']
    alphas = [0.01,  0.1, 0.3]

    for attack in attacks:
        for alpha in alphas:
            for target in targets:
                print(attack, alpha, target)
                count(attack, alpha, target)

