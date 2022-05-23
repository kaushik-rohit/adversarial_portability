import torchvision.transforms as transforms
from shared import model_names, get_model
from ImageNetWithPaths import ImageNetWithPaths
import torch.multiprocessing as mp
from tqdm import tqdm
import torch
import os

# constants
imagenet_path = '/local/scratch/rohit/datasets/ImageNet/ILSVRC2012'
output_path = '/local/scratch/rohit/datasets/ImageNet/ILSVRC2012/adv_ex/{}/{}/{}/train'
alphas = [0.01]
attacks = ['PGD']
BATCH_SIZE = 4
gpu_ids = [1, 4, 5, 6]
n_GPU = len(gpu_ids)
pbar = tqdm(total=len(model_names) * 10000 / (BATCH_SIZE * n_GPU))

def verify_count():
    for model_name in model_names:
        for attack in attacks:
            for alpha in alphas:
                print('verifying {} {} {}'.format(model_name, attack, alpha))
                path = output_path.format(model_name, attack, alpha)
                classes = os.listdir(path)
                assert(len(classes) == 1000)
                for _class in classes:
                    class_path = os.path.join(path, _class)
                    figs = os.listdir(class_path)
                    assert len(figs) == 10, "{}".format(class_path)


def verify_subset_data(rank, model_name):
    model = get_model(model_name)
    model.eval()
    model.to(rank)

    # load data
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                        std=[0.229, 0.224, 0.225])
                                        
    data = ImageNetWithPaths(imagenet_path, split='train', 
                            transform=transforms.Compose([
                                transforms.Resize(256),
                                transforms.CenterCrop(224),
                                transforms.ToTensor(),
                                normalize
                                ]))

    data_loader = torch.utils.data.DataLoader(data,
                                            batch_size=BATCH_SIZE,
                                            shuffle=True,
                                            num_workers=0)


    for data in data_loader:
        cln_data, true_label, paths = data
        cln_data = cln_data.to(rank)
        true_label = true_label.to(rank)

        preds = model(cln_data)
        preds = torch.argmax(preds, dim=1)
        # print(torch.all(preds == true_label).item())
        assert(torch.all(preds == true_label).item())
        pbar.update(1)

def verify_subset_data_helper():
    pool = mp.Pool(n_GPU)
    pool.starmap(verify_subset_data, [(gpu_ids[i % n_GPU], model_names[i]) for i in range(len(model_names))])
    pool.close()

if __name__ == '__main__':
    mp.set_start_method('spawn', force=True)
    verify_count()
    # verify_subset_data_helper()
