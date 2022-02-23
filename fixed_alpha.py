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
attacks = ['FGV']
gpu_ids = [0, 2]
n_GPU = len(gpu_ids)
pbar = tqdm(total= len(model_names) * len(model_names) * int(10000 / (BATCH_SIZE*n_GPU)))

columns = ['target'] + model_names

def pytorch_loader(filename):
    return torch.load(filename)

def is_valid_file(filepath):
    return True

def classification_on_target(rank, target, attack, alpha):
    model = get_model(target)
    model.eval()
    model.to(rank)

    row = [target]
    for source in model_names:
        print('portability for source {} and target {} on gpu {} with attack {} and alpha {}'.format(source, target, rank, attack, alpha))

        path_to_adv_ex = '/local/scratch/rohit/datasets/ImageNet/ILSVRC2012/adv_ex/{}/{}/{}'.format(source, attack, alpha)

        # load data
        normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                        std=[0.229, 0.224, 0.225])

        data = ImageNetWithPaths(path_to_adv_ex, split='train', loader=pytorch_loader, is_valid_file=is_valid_file)

        data_loader = torch.utils.data.DataLoader(data,
                                                batch_size=BATCH_SIZE,
                                                shuffle=True,
                                                num_workers=0)

        incorrectly_labeled_adv_samples = 0

        for data in data_loader:
            cln_data, true_label, _ = data
            cln_data = cln_data.to(rank)
            true_label = true_label.to(rank)

            preds = model(cln_data)
            preds = torch.argmax(preds, dim=1)
            n = preds.size(dim=0) - torch.sum(preds == true_label).item()
            incorrectly_labeled_adv_samples += n
            pbar.update(1)
        row.append(incorrectly_labeled_adv_samples)

    return row

def test_portability_for_examples_on_target(attack, alpha):
    pool = mp.Pool(n_GPU)
    rows = pool.starmap(classification_on_target, [(gpu_ids[i % n_GPU], model_names[i], attack, alpha) for i in range(len(model_names))])
    pool.close()
    
    return rows

def caculate_adversary_with_fixed_alpha():
    for attack in attacks:
        for alpha in alphas:
            rows = test_portability_for_examples_on_target(attack, alpha)
            res_df = pd.DataFrame(rows, columns=columns)
            res_df.to_csv('{}_alpha={}.csv'.format(attack, alpha), index=False)

if __name__ == '__main__':
    caculate_adversary_with_fixed_alpha()
