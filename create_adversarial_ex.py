import torchvision.transforms as transforms
from advertorch.attacks import PGDAttack, FGSM, FGV
from advertorch.attacks import CarliniWagnerL2Attack as CRW
from ImageNetWithPaths import ImageNetWithPaths
import torch.multiprocessing as mp
from torchvision.utils import save_image
import shared
from shared import model_names, get_model, get_adversarial_model
from tqdm import tqdm
import torch
import pathlib
import os

# constants
imagenet_path = '/local/scratch/rohit/datasets/ImageNet/ILSVRC2012'
output_path = '/local/scratch/rohit/datasets/ImageNet/ILSVRC2012/adv_ex/{}/{}/{}/train/{}'
alphas = [0.01]
attacks = ['FGV']
BATCH_SIZE = 4
gpu_ids = [0, 2]
n_GPU = len(gpu_ids)
pbar = tqdm(total=len(model_names) * 10000 / (BATCH_SIZE * n_GPU))
def generate_adversarial_example(rank, model_name):
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

    adversarial_models = {attack: {} for attack in attacks}

    for attack in attacks:
        for alpha in alphas:
            adversarial_models[attack][alpha] = get_adversarial_model(model, attack, alpha)

    for data in data_loader:
        cln_data, true_label, paths = data
        cln_data = cln_data.to(rank)
        true_label = true_label.to(rank)
        for attack in attacks:
            for alpha in alphas:
                adv_model = adversarial_models[attack][alpha]
                adv_untargeted = adv_model.perturb(cln_data, true_label)

                for batch in range(BATCH_SIZE):
                    class_name = paths[batch].split('/')[-2]
                    file_name = paths[batch].split('/')[-1].split('.JPEG')[0] + '.pt'
                    save_path = output_path.format(model_name, attack, alpha, class_name)
                    pathlib.Path(save_path).mkdir(parents=True, exist_ok=True)
                    save_file = os.path.join(save_path, file_name)
                    torch.save(adv_untargeted[batch, :, :, :], save_file)

        pbar.update(1)


if __name__ == '__main__':
    mp.set_start_method('spawn', force=True)
    pool = mp.Pool(n_GPU)
    pool.starmap(generate_adversarial_example, [(gpu_ids[i % n_GPU], model_names[i]) for i in range(len(model_names))])
    pool.close()
