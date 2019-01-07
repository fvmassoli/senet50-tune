#################
# Python imports
#################
import argparse

##################
# Pytorch imports
##################
import torch
import torch.nn as nn
import torchvision.transforms as t
import torchvision.models as models
from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder
from torch.utils.data.sampler import WeightedRandomSampler


def get_args():
    parser = argparse.ArgumentParser(description='hyp_search_tune')
    parser.add_argument('-s', '--seed', type=int, default=41, help='Set the randome generators seed (default: 41)')
    parser.add_argument('-c', '--numCpu', type=int, default=4, help='Set the number of CPU to use (default: 4)')
    parser.add_argument('-g', '--numGpu', type=int, default=0, help='Set the number of GPU to use (default: 0)')
    parser.add_argument('-ns', '--numSamples', type=int, default=1,
                        help='Number of experiment configurations to be run (default: 1)')
    parser.add_argument('-ti', '--trainingIteration', type=int, default=1,
                        help='Number of training iterations for each experiment configuration (default: 1)')
    parser.add_argument('-lf', '--logFrequency', type=int, default=10,
                        help='Frequency (unit=single train iteration) for log train/valid stats')
    parser.add_argument('-cf', '--checkpointFreq', type=int, default=0,
                        help='Frequency (unit=iteration) to checkpoint the model (default: 0 --- i.e. disabled)')
    parser.add_argument('-t', '--runTensorBoard', action='store_true', help='Run tensorboard (default: false)')
    parser.add_argument('-df', '--dataFolder', help='Path to main data folder')
    return parser.parse_args()


def make_weights_for_balanced_classes(images, n_classes):
    """
    Since the classes are not balanced let's create weights for them

    """
    count = [0] * n_classes
    for item in images:
        count[item[1]] += 1
    weight_per_class = [0.] * n_classes
    N = float(sum(count))
    for i in range(n_classes):
        weight_per_class[i] = N / float(count[i])
    weights = [0] * len(images)
    for idx, val in enumerate(images):
        weights[idx] = weight_per_class[val[1]]
    return weights


def get_transforms(train, gs=0.2, mean=[]):
    if train:
        return t.Compose([
            t.Resize(256),
            t.RandomGrayscale(p=gs),
            t.RandomCrop(224),
            t.ToTensor(),

        ])
    else:
        return t.Compose([
            t.Resize(256),
            t.CenterCrop(224),
            t.ToTensor(),

        ])


def get_loaders(train_batch_size, num_workers=1, data_folder=None, cuda_available=False):
    print("Loading data...")
    train_data_set = ImageFolder(root=data_folder+'/train', transform=get_transforms(train=True))
    valid_data_set = ImageFolder(root=data_folder+'/valid', transform=get_transforms(train=False))
    weights = make_weights_for_balanced_classes(train_data_set.imgs, len(train_data_set.classes))
    weights = torch.DoubleTensor(weights)
    sampler = WeightedRandomSampler(weights, len(weights))
    train_data_loader = DataLoader(dataset=train_data_set,
                                   sampler=sampler,
                                   batch_size=train_batch_size,
                                   num_workers=num_workers,
                                   pin_memory=cuda_available)
    valid_data_loader = DataLoader(dataset=valid_data_set,
                                   batch_size=8,
                                   num_workers=num_workers,
                                   pin_memory=cuda_available)
    print("Data loaded!!!")
    return train_data_loader, valid_data_loader


def get_model():
    import importlib
    print("Loading model...")
    MainModel = importlib.load_source('MainModel', './models/senet50_ft_pytorch/senet50_ft_pytorch.py')
    model = torch.load('./models/senet50_ft_pytorch/senet50_ft_pytorch.pth')
    print("Model loaded!!!")
    return model


def launch_tensorboard(logdir):
    import os
    os.system('tensorboard --logdir=~/ray_results/' + logdir)
    return
