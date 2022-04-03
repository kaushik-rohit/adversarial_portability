import torch
import torchvision.transforms as transforms
from ImageNetWithPaths import ImageNetWithPaths
import torch.multiprocessing as mp
import pandas as pd
from tqdm import tqdm
from shared import model_names, get_model

# constants
BATCH_SIZE = 4
alphas = [0.01]
attacks = ['FGSM']
gpu_ids = [3, 6, 7]
n_GPU = len(gpu_ids)
pbar = tqdm(total= len(model_names) * len(model_names) * int(10000 / (BATCH_SIZE*n_GPU)))

columns = ['image', 'target'] + model_names + ['true_label']

def pytorch_loader(filename):
    return torch.load(filename)

def is_valid_file(filepath):
    return True

def classification_on_target(rank, target, attack, alpha):
    model = get_model(target)
    model.eval()
    model.to(rank)

    predictions = {}
    true_labels = {}

    for source in model_names:
        print('portability for source {} and target {} on gpu {} with attack {} and alpha {}'.format(source, target, rank, attack, alpha))

        path_to_adv_ex = '/local/scratch/rohit/datasets/ImageNet/ILSVRC2012/adv_ex/{}/{}/{}'.format(source, attack, alpha)

        # load data
        data = ImageNetWithPaths(path_to_adv_ex, split='train', loader=pytorch_loader, is_valid_file=is_valid_file)

        data_loader = torch.utils.data.DataLoader(data,
                                                batch_size=BATCH_SIZE,
                                                shuffle=True,
                                                num_workers=0)

        incorrectly_labeled_adv_samples = 0

        for data in data_loader:
            cln_data, true_label, paths = data
            cln_data = cln_data.to(rank)
            true_label = true_label.to(rank)

            preds = model(cln_data)
            preds = torch.argmax(preds, dim=1)
            n = preds.size(dim=0) - torch.sum(preds == true_label).item()
            incorrectly_labeled_adv_samples += n
            pbar.update(1)

            for batch in range(BATCH_SIZE):
                file_name = paths[batch].split('/')[-1].split('.JPEG')[0]

                if file_name not in true_labels:
                    true_labels[file_name] = true_label[batch].item()

                if file_name in predictions:
                    predictions[file_name][source] = preds[batch].item()
                else:
                    predictions[file_name] = {s: 0 for s in model_names}
                    predictions[file_name][source] = preds[batch].item()
    rows = []

    for file_name in predictions.keys():
        row = [file_name, target]
        for source in model_names:
            row.append(predictions[file_name][source])
        row.append(true_labels[file_name])
        rows.append(row)

    return rows

def test_portability_for_examples_on_target(attack, alpha):
    pool = mp.Pool(n_GPU)
    rows = pool.starmap(classification_on_target, [(gpu_ids[i % n_GPU], model_names[i], attack, alpha) for i in range(len(model_names))])
    pool.close()
    
    return rows

def caculate_adversary_with_fixed_alpha():
    for attack in attacks:
        for alpha in alphas:
            rows = test_portability_for_examples_on_target(attack, alpha)
            rows = [x for sublist in rows for x in sublist]
            # rows = classification_on_target(gpu_ids[0], model_names[0], attack, alpha)
            res_df = pd.DataFrame(rows, columns=columns)
            res_df.to_csv('{}_alpha={}_preds.csv'.format(attack, alpha), index=False)

if __name__ == '__main__':
    caculate_adversary_with_fixed_alpha()
