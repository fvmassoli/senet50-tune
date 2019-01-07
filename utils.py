##################
# Pytorch imports
##################
import torch

#################
# Python imports
#################
import argparse


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
    return parser.parse_args()


def launch_tensorboard(logdir):
    import os
    os.system('tensorboard --logdir=~/ray_results/' + logdir)
    return


def get_model():
    import importlib
    print("Loading model...")
    MainModel = importlib.load_source('MainModel', './models/senet50_ft_pytorch/senet50_ft_pytorch.py')
    model = torch.load('./models/senet50_ft_pytorch/senet50_ft_pytorch.pth')
    print("Model loaded!!!")
    return model
