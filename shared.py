import torchvision
from advertorch.attacks import PGDAttack, FGSM, FGV
from advertorch.attacks import CarliniWagnerL2Attack as CRW
from anytree import Node
from anytree.util import commonancestors
import torch
import torch.nn as nn
import pandas as pd

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


class Normalize(nn.Module):
    def __init__(self, mean, std) :
        super(Normalize, self).__init__()
        self.register_buffer('mean', torch.Tensor(mean))
        self.register_buffer('std', torch.Tensor(std))
        
    def forward(self, input):
        # Broadcasting
        mean = self.mean.reshape(1, 3, 1, 1)
        std = self.std.reshape(1, 3, 1, 1)
        return (input - mean) / std

norm_layer = Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])


class ImageNetHeirarchy():
    def __init__(self, path, wnid_to_label):
        self.nodes = {}
        self.tree = None
        self.label_to_wnid = {}
        self._populate_tree(path)
        self._get_label_map(wnid_to_label)

    def _get_label_map(self, wnid_to_label):
        df = pd.read_csv(wnid_to_label)
        for _, row in df.iterrows():
            self.label_to_wnid[row['label']] = row['wnid']

    def _populate_tree(self, path):
        with open(path, 'r') as f:
            lines = f.readlines()

        for line in lines:
            parent, child = line.split()
            parent = parent.strip()
            child = child.strip()
            if parent not in self.nodes:
                self.nodes[parent] = Node(parent)
                self.tree = self.nodes[parent]
            self.nodes[child] = Node(child, parent=self.nodes[parent])

    def get_node(self, wnid):
        return self.nodes[wnid]

    def get_lca(self, l1, l2):
        wnid1 = self._get_wnid(l1)
        wnid2 = self._get_wnid(l2)

        node1 = self.get_node(wnid1)
        node2 = self.get_node(wnid2)
        lca = commonancestors(node1, node2)[-1]

        cost = node1.depth - lca.depth + node2.depth - lca.depth
        return cost

    def _get_wnid(self, label):
        return self.label_to_wnid[label]


def get_model(model_name):
    if model_name == 'resnet18':
        model = nn.Sequential(
            norm_layer,
            torchvision.models.resnet18(pretrained=True)
        )
    elif model_name == 'resnet34':
        model = nn.Sequential(
            norm_layer,
            torchvision.models.resnet34(pretrained=True)
        )
    elif model_name == 'resnet50':
        model = nn.Sequential(
            norm_layer,
            torchvision.models.resnet50(pretrained=True)
        )
    elif model_name == 'resnet101':
        model = nn.Sequential(
            norm_layer,
            torchvision.models.resnet101(pretrained=True)
        )
    elif model_name == 'resnet152':
        model = nn.Sequential(
            norm_layer,
            torchvision.models.resnet152(pretrained=True)
        )
    elif model_name == 'vgg16':
        model = nn.Sequential(
            norm_layer,
            torchvision.models.vgg16(pretrained=True)
        )
    elif model_name == 'vgg19':
        model = nn.Sequential(
            norm_layer,
            torchvision.models.vgg19(pretrained=True)
        )
    elif model_name == 'googlenet':
        model = nn.Sequential(
            norm_layer,
            torchvision.models.googlenet(pretrained=True)
        )
    elif model_name == 'densenet':
        model = nn.Sequential(
            norm_layer,
            torchvision.models.densenet161(pretrained=True)
        )
    elif model_name == 'mobilenet_v2':
        model = nn.Sequential(
            norm_layer,
            torchvision.models.mobilenet_v2(pretrained=True)
        )
    elif model_name == 'mobilenet_v3_small':
        model = nn.Sequential(
            norm_layer,
            torchvision.models.mobilenet_v3_small(pretrained=True)
        )
    elif model_name == 'mobilenet_v3_large':
        model = nn.Sequential(
            norm_layer,
            torchvision.models.mobilenet_v3_large(pretrained=True)
        )
    elif model_name == 'wide_resnet_50_2':
        model = nn.Sequential(
            norm_layer,
            torchvision.models.wide_resnet50_2(pretrained=True)
        )

    return model

def get_adversarial_model(model, attack, alpha, clip_min=None, clip_max=None):
    if attack == 'PGD':
        adv_model = PGDAttack(model, loss_fn=torch.nn.CrossEntropyLoss(reduction='sum'),
                    eps=alpha, rand_init=True, clip_min=0, clip_max=1, targeted=False)
    elif attack == 'FGSM':
        adv_model = FGSM(model, loss_fn=torch.nn.CrossEntropyLoss(reduction='sum'), clip_min=0, clip_max=1, 
                        eps=alpha, targeted=False)
    elif attack == 'CRW':
        adv_model = CRW(model, 1000, learning_rate=alpha)
    elif attack == 'FGV':
        adv_model = FGV(model, loss_fn=torch.nn.CrossEntropyLoss(reduction='sum'), eps=alpha, targeted=False)

    return adv_model
