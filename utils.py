##################
# Pytorch imports
##################
import torch
import torch.nn as nn

#################
# Python imports
#################
import os
import argparse
import importlib


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
    parser.add_argument('-tf', '--trainFolder', help='Training folder path')
    parser.add_argument('-vf', '--validFolder', help='Validation folder path')
    parser.add_argument('-ir', '--imageResolution', default=32, help='Image resolution (default: 32)')
    parser.add_argument('-p', '--lowerResolutionProb', default=0.5, type=float, help='Lower resolution prob(default: 0.5)')
    parser.add_argument('-is', '--indicesStep', type=int, default=100, help='Indices step (default: 100)')
    parser.add_argument('-spt', '--trainValidSplit', default=0.1, type=float, help='Indices step (default: 0.1)')
    return parser.parse_args()


def launch_tensorboard(logdir):
    os.system('tensorboard --logdir=~/ray_results/' + logdir)
    return


def get_model():
    print("Loading model...")
    b = '/home/fabiom/faces/vgg_face_2/pytorch_model/senet50-tune'
#    MainModel = importlib.load_source('MainModel', os.path.joi(b, 'senet50_ft_pytorch.py'))
    model = torch.load(os.path.join(b, 'senet50_ft_pytorch.pth'))
    for n, m in model.named_modules():
        if isinstance(m, nn.BatchNorm2d):
            m.momentum = 0.1
            m.eps = 1e-05
#    print(model)
    print("Model loaded!!!")
    return model
