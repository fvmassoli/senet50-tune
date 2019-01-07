from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader, Subset

from .utils import *
from .face_dataset import VggFace2Dataset


class DataManager(object):
    def __init__(self,
                 train_folder,
                 valid_folder,
                 train_batch_size,
                 valid_batch_size,
                 img_resolution,
                 interpolation_algo_name,
                 interpolation_algo_val,
                 lowering_resolution_prob,
                 indices_step,
                 training_valid_split):
        self.train_folder = train_folder
        self.valid_folder = valid_folder
        self.train_batch_size = train_batch_size
        self.valid_batch_size = valid_batch_size
        self.img_resolution = img_resolution
        self.interpolation_algo_name = interpolation_algo_name
        self.interpolation_algo_val = interpolation_algo_val
        self.lowering_resolution_prob = lowering_resolution_prob
        self.indices_step = indices_step
        self.training_valid_split = training_valid_split
        self.train_data_set, self.valid_data_set_lr, self.valid_data_set = self._init_data_sets()
        self.train_data_loader, self.valid_data_loader, self.valid_data_loader_original_data = self._init_data_loaders()
        self._print_data_summary()

    def _init_data_sets(self):
        print("Loading data ... ")
        train_data_set = VggFace2Dataset(root=self.train_folder,
                                         resolution=self.img_resolution,
                                         algo_name=self.interpolation_algo_name,
                                         algo_val=self.interpolation_algo_val,
                                         transforms=get_train_transforms(resize=256, grayed_prob=0.2, crop_size=224),
                                         lowering_resolution_prob=self.lowering_resolution_prob)
        valid_data_set_lr = VggFace2Dataset(root=self.valid_folder,
                                            resolution=self.img_resolution,
                                            algo_name=self.interpolation_algo_name,
                                            algo_val=self.interpolation_algo_val,
                                            transforms=get_valid_transforms(resize=256, crop_size=224),
                                            lowering_resolution_prob=1.0)
        valid_data_set = ImageFolder(root=self.lower_resolution_validation_folder,
                                     transform=get_valid_transforms(resize=256, crop_size=224))
        print("Data loaded")
        return train_data_set, valid_data_set_lr, valid_data_set

    def _init_data_loaders(self):
        dataset_len = len(self.train_data_set)
        indices = list(np.arange(0, dataset_len, self.indices_step))
        split = int(np.floor(len(indices) * self.training_valid_split))
        train_indices = indices[split:]

        dataset_len = len(self.valid_data_set)
        indices = list(np.arange(0, dataset_len, self.indices_step))
        split = int(np.floor(len(indices) * self.training_valid_split))
        valid_indices = indices[split:]

        # Get data subsets
        tmp_train_data_set = Subset(self.train_data_set, train_indices)
        tmp_valid_data_set_lr = Subset(self.self.valid_data_set_lr, valid_indices)
        tmp_valid_data_set_original_data = Subset(self.valid_data_set, valid_indices)
        # Create data loaders
        train_data_loader = DataLoader(dataset=tmp_train_data_set,
                                       batch_size=self.train_batch_size,
                                       num_workers=4,
                                       pin_memory=torch.cuda.is_available())
        valid_data_loader_lr = DataLoader(dataset=tmp_valid_data_set_lr,
                                          batch_size=self.valid_batch_size,
                                          num_workers=4,
                                          pin_memory=torch.cuda.is_available())
        valid_data_loader_original_data = DataLoader(dataset=tmp_valid_data_set_original_data,
                                                     batch_size=self.valid_batch_size,
                                                     num_workers=4,
                                                     pin_memory=torch.cuda.is_available())
        return train_data_loader, valid_data_loader_lr, valid_data_loader_original_data

    def get_data_sets(self):
        return self.train_data_set, self.valid_data_set, self.valid_data_set_original_data

    def get_data_loaders(self):
        return self.train_data_loader, self.valid_data_loader, self.valid_data_loader_original_data

    def _print_data_summary(self):
        print('"' * 50)
        print('"' * 50)
        print("Data summary:"
              "\n Number of classes:            {}"
              "\n Number of training images:    {}"
              "\n Number of validation images:  {}"
              "\n Number of training batches:   {}"
              "\n Number of validation batches: {}"
              .format(len(self.train_data_set.classes),
                      len(self.train_data_loader)*self.train_batch_size,
                      len(self.valid_data_loader)*self.valid_batch_size,
                      len(self.train_data_loader),
                      len(self.valid_data_loader)))
        print('"' * 50)
        print('"' * 50)
