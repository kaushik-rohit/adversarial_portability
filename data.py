import torch
import torchvision.transforms as transforms
import pandas as pd
from ImageNetWithPaths import ImageNetWithPaths
import shared
from tqdm import tqdm
from itertools import cycle
import pickle

imagenet_path = '/local/scratch/datasets/ImageNet/ILSVRC2012'

# constants
MAX_ITER = 10000
BATCH_SIZE = 1
N_CLASS = 1000
IMG_PER_CLASS = 10
device_ids = [0, 1, 2, 3, 4, 5, 6]
devices = cycle(device_ids)
pbar = tqdm(total=10000)

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
                                          num_workers=4)

models = {}
# move model to gpus
for model_name in shared.model_names:
    device = next(devices)
    models[model_name] = shared.get_model(model_name)
    models[model_name].to(device)

# set models to eval mode
for model_name in shared.model_names:
    models[model_name].eval()

class_paths = {i: [] for i in range(N_CLASS)}

def check_data(paths):
    lens = list(map(len, paths))

    if lens.count(IMG_PER_CLASS) == len(lens):
        return True
    
    return False


for data in data_loader:
    cln_data, true_label, path = data
    class_id = true_label.item()
    flag = True

    if len(class_paths[class_id]) == IMG_PER_CLASS:
        continue
    
    # print('true_label ', true_label)
    for model_name in shared.model_names:
        cln_data = cln_data.to(next(models[model_name].parameters()).device)
        true_label = true_label.to(next(models[model_name].parameters()).device)
        pred = models[model_name](cln_data)
        pred = torch.argmax(pred, dim=1)
        print(true_label, pred)
        if not torch.eq(pred, true_label):
            flag = False
            break
    if flag:
        # print(path)
        class_paths[class_id].append(path)
        pbar.update(1)
    
    if check_data(class_paths.values()):
        break

with open('dataset.pkl', 'wb') as f:
    pickle.dump(class_paths, f)
