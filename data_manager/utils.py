#################
# Python imports
#################
import numpy as np

##################
# Pytorch imports
##################
import torchvision.transforms as t


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


def get_valid_transforms(resize=256, crop_size=224, mean_vector=[131.0912, 103.8827, 91.4953]):
    return t.Compose(
        [
            t.Resize(resize),
            t.CenterCrop(crop_size),
            t.ToTensor(),
            t.Lambda(lambda x: subtract_mean(x, mean_vector))
        ]
    )
