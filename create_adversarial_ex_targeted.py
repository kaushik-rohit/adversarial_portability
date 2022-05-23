import torchvision.transforms as transforms
from ImageNetWithPaths import ImageNetWithPaths
import torch.multiprocessing as mp
from shared import model_names, get_model, get_adversarial_model
from tqdm import tqdm
import numpy as np
import torch
import pathlib
import pickle
import os

# constants
target_option = 'median' # target selection criterai could be median, closest, farthest
imagenet_path = '/local/scratch/rohit/datasets/ImageNet/ILSVRC2012'
output_path = '/local/scratch/rohit/datasets/ImageNet/ILSVRC2012/adv_target_ex/{}/{}/{}/{}/train/{}'
cost_matrix_path = '/home/user/rohit/adversarial-attacks/data/cost_matrix.pkl'

alphas = [0.01, 0.1, 0.3]
attacks = ['FGSM', 'FGV', 'PGD']
BATCH_SIZE = 4
gpu_ids = [1, 2, 3, 4, 5, 6, 7] # ids of the gpus to be used

n_GPU = len(gpu_ids)
pbar = tqdm(total=len(model_names) * 10000 / (BATCH_SIZE * n_GPU))


def generate_adversarial_example(rank, model_name):
    '''
    @rank: id of the gpu to be used
    @model name: name of the neural network model to be used as a source
    
    generates targeted adversarial example with the specified attacks and target selection
    criteria.
    '''
    print('generating adversarial examples using {} on GPU {}'.format(model_name, rank))

    model = get_model(model_name)
    model.eval()
    model.to(rank)

    # load data

    data = ImageNetWithPaths(imagenet_path, split='train',
                            transform=transforms.Compose([
                                transforms.Resize(256),
                                transforms.CenterCrop(224),
                                transforms.ToTensor()
                                ]))

    data_loader = torch.utils.data.DataLoader(data,
                                            batch_size=BATCH_SIZE,
                                            shuffle=True,
                                            num_workers=0)
    
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

    adversarial_models = {attack: {} for attack in attacks}

    for attack in attacks:
        for alpha in alphas:
            adversarial_models[attack][alpha] = get_adversarial_model(model, attack, alpha, targeted=True)

    for data in data_loader:
        cln_data, true_label, paths = data
        cln_data = cln_data.to(rank)
        target_label = true_label.apply_(lambda x: get_target_label(x))
        target_label = target_label.to(rank)

        for attack in attacks:
            for alpha in alphas:
                adv_model = adversarial_models[attack][alpha]
                adv_untargeted = adv_model.perturb(cln_data, target_label)

                for batch in range(BATCH_SIZE):
                    class_name = paths[batch].split('/')[-2]
                    file_name = paths[batch].split('/')[-1].split('.JPEG')[0] + '.pt'
                    save_path = output_path.format(target_option, model_name, attack, alpha, class_name)
                    pathlib.Path(save_path).mkdir(parents=True, exist_ok=True)
                    save_file = os.path.join(save_path, file_name)
                    torch.save(adv_untargeted[batch, :, :, :], save_file)

        pbar.update(1)


if __name__ == '__main__':
    mp.set_start_method('spawn', force=True)
    pool = mp.Pool(n_GPU)
    pool.starmap(generate_adversarial_example, [(gpu_ids[i % n_GPU], model_names[i]) for i in range(len(model_names))])
    pool.close()
