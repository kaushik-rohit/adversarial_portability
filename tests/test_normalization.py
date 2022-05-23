import torchvision.transforms as transforms
from advertorch.attacks import PGDAttack, FGSM, FGV
from advertorch.attacks import CarliniWagnerL2Attack as CRW
from ImageNetWithPaths import ImageNetWithPaths
import torch.multiprocessing as mp
from torchvision.utils import save_image
from shared import model_names, get_model, get_adversarial_model
from tqdm import tqdm
import torch
import pathlib
import os

# constants
imagenet_path = '/local/scratch/rohit/datasets/ImageNet/ILSVRC2012'
output_path = '/local/scratch/rohit/datasets/ImageNet/ILSVRC2012/adv_ex/{}/{}/{}/train/{}'
alphas = [0.3]
attacks = ['FGSM']
BATCH_SIZE = 1
gpu_ids = [4, 5, 6]
n_GPU = len(gpu_ids)

def generate_adversarial_example(rank, model_name):
    print('generating adversarial examples using {} on GPU {}'.format(model_name, rank))

    model = get_model(model_name)
    model.eval()
    model.to(rank)

    # load data
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                        std=[0.229, 0.224, 0.225])
                                        
    data = ImageNetWithPaths(imagenet_path, split='train', 
                            transform=transforms.Compose([
                                transforms.RandomResizedCrop(224),
                                transforms.ToTensor(),
                                normalize
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
        print('original image: ', torch.min(cln_data, dim=1)[0], torch.min(cln_data, dim=2)[0])

        for attack in attacks:
            for alpha in alphas:
                adv_model = adversarial_models[attack][alpha]
                adv_untargeted = adv_model.perturb(cln_data, true_label)

                print('adv example: ', torch.min(adv_untargeted, dim=1)[0], torch.min(adv_untargeted, dim=2)[0])


def classification_on_target(rank, target, attack, alpha):
    model = get_model(target)
    model.eval()
    model.to(rank)

    row = [target]
    print('validating examples created on {} on gpu {} with attack {} and alpha {}'.format(target, rank, attack, alpha))

    path_to_adv_ex = '/local/scratch/rohit/datasets/ImageNet/ILSVRC2012/adv_ex/{}/{}/{}'.format(target, attack, alpha)

    # load data
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                    std=[0.229, 0.224, 0.225])

    data = ImageNetWithPaths(path_to_adv_ex, split='train', transform=transforms.Compose([transforms.ToTensor()]))

    data_loader = torch.utils.data.DataLoader(data,
                                            batch_size=BATCH_SIZE,
                                            shuffle=True,
                                            num_workers=0)

    for data in data_loader:
        cln_data, true_label, _ = data
        cln_data = cln_data.to(rank)
        true_label = true_label.to(rank)
        print('original image: ', torch.min(cln_data).item(), torch.max(cln_data).item())

    return row

if __name__ == '__main__':
    # classification_on_target(gpu_ids[0], model_names[0], attacks[0], alphas[0])
    generate_adversarial_example(gpu_ids[0], model_names[0])
