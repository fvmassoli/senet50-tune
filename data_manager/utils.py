import os
import numpy as np
from PIL import Image
from tqdm import tqdm

import torch
import torchvision.transforms as t


def _get_train_sampler(self):
    images = self.train_data_set.samples
    n_classes = len(self.train_data_set.classes)
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
    weights = torch.DoubleTensor(weights)
    return torch.utils.data.sampler.WeightedRandomSampler(weights, len(weights))


def subtract_mean(x, mean_vector):
    x *= 255.
    if x.shape[0] == 1:
        x = np.tile(3, 1, 1)
    x[0] -= mean_vector[0]
    x[1] -= mean_vector[1]
    x[2] -= mean_vector[2]
    return x


def get_train_transforms(resize=256, grayed_prob=0.2, crop_size=224, mean_vector=[131.0912, 103.8827, 91.4953]):
    return t.Compose(
        [
            t.Resize(resize),
            t.RandomGrayscale(p=grayed_prob),
            t.RandomCrop(crop_size),
            t.ToTensor(),
            t.Lambda(lambda x: subtract_mean(x, mean_vector))
        ]
    )


def get_valid_transforms(resize=256,crop_size=224, mean_vector=[131.0912, 103.8827, 91.4953]):
    return t.Compose(
        [
            t.Resize(resize),
            t.CenterCrop(crop_size),
            t.ToTensor(),
            t.Lambda(lambda x: subtract_mean(x, mean_vector))
        ]
    )


def lower_resolution(img, algo, resolution):
    w_i, h_i = img.size
    r = h_i/float(w_i)
    if h_i<w_i:
        h_n = resolution
        w_n = h_n/float(r)
    else:
        w_n = resolution
        h_n = w_n*float(r)
    img2 = img.resize((int(w_n), int(h_n)), algo)
    img2 = img2.resize((w_i, h_i), algo)
    return img2


def create_validation_dataset_for_lower_resolution(path, folder, interpolation_algo_val, resolution):
    image_output_folder = path
    desc = 'Creating validation folder with resolution: '+str(resolution)
    for n_ in tqdm(os.listdir(folder), desc=desc):
        f = os.path.join(image_output_folder, n_)
        if not os.path.isdir(f):
            os.makedirs(f)
        for img_n in os.listdir(os.path.join(folder, n_)):
            img = Image.open(os.path.join(folder, n_, img_n))
            img.convert('RGB')
            # if torch.rand(1).item() < lowering_resolution_prob:
            img = lower_resolution(img, interpolation_algo_val, resolution)
            f_name = os.path.join(image_output_folder, n_, img_n)
            img.save(f_name)
    return image_output_folder
