from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader, Subset

from utils import *
from face_dataset import VggFace2Dataset


class DataManager(object):
    def __init__(self,
                 train_folder,
                 valid_folder,
                 create_lower_resolution_validation_folder=False,
                 train_batch_size=1,
                 valid_batch_size=1,
                 img_resolution=128,
                 interpolation_algo_name=None,
                 lowering_resolution_prob=0,
                 test_on_small_data_set=False,
                 indices_step=100,
                 training_valid_split=0.1):
        self.train_folder = train_folder
        self.valid_folder = valid_folder
        self.train_batch_size = train_batch_size
        self.valid_batch_size = valid_batch_size
        self.img_resolution = img_resolution
        self.interpolation_algo_name = interpolation_algo_name
        self.lowering_resolution_prob = lowering_resolution_prob
        self.test_on_small_data_set = test_on_small_data_set
        self.indices_step = indices_step
        self.training_valid_split = training_valid_split
        self.create_lower_resolution_validation_folder = create_lower_resolution_validation_folder
        self.lower_resolution_validation_folder = self._create_lower_resolution_validation_folder()
        self.train_data_set, self.valid_data_set, self.valid_data_set_original_data = self._init_data_sets()
        self.train_data_loader, self.valid_data_loader, self.valid_data_loader_original_data = self._init_data_loaders()
        self._print_data_summary()

    def _create_lower_resolution_validation_folder(self):
        base = '/ssd/fabiom/validation_lower_resolution/validation_'
        if self.create_lower_resolution_validation_folder:
            return create_validation_dataset_for_lower_resolution(base=base,
                                                                  folder=self.valid_folder,
                                                                  algo_name=self.interpolation_algo_name,
                                                                  resolution=self.img_resolution,
                                                                  lowering_resolution_prob=self.lowering_resolution_prob),
        else:
            return base+self.interpolation_algo_name+'_'+str(self.img_resolution)+'_'+str(1.1)

    def _init_data_sets(self):
        print("Loading data ... ")
        train_data_set = VggFace2Dataset(root=self.train_folder,
                                         resolution=self.img_resolution,
                                         algo_name=self.interpolation_algo_name,
                                         transforms=get_train_transforms(resize=256, grayed_prob=0.2, crop_size=224),
                                         lowering_resolution_prob=self.lowering_resolution_prob)
        valid_data_set = ImageFolder(root=self.lower_resolution_validation_folder,
                                     transform=get_valid_transforms(resize=256, crop_size=224))
        valid_data_set_original_data = ImageFolder(root=self.valid_folder,
                                                   transform=get_valid_transforms(resize=256, crop_size=224))
        print("Data loaded")
        return train_data_set, valid_data_set, valid_data_set_original_data

    def _init_data_loaders(self):
        if self.test_on_small_data_set:
            return self._init_data_loaders_for_overfit_test()
        else:
            return self._init_full_data_loaders()

    def _init_data_loaders_for_overfit_test(self):
        dataset_len = len(self.train_data_set)
        indices = list(np.arange(0, dataset_len, self.indices_step))
        split = int(np.floor(len(indices) * self.training_valid_split))
        train_indices = indices[split:]
        valid_indices = indices[:split]
        # Get data subsets
        tmp_train_data_set = Subset(self.train_data_set, train_indices)
        tmp_valid_data_set = Subset(self.valid_data_set, valid_indices)
        tmp_valid_data_set_original_data = Subset(self.valid_data_set_original_data, valid_indices)
        # Create data loaders
        train_data_loader = DataLoader(dataset=tmp_train_data_set,
                                       batch_size=self.train_batch_size,
                                       num_workers=4,
                                       pin_memory=torch.cuda.is_available())
        valid_data_loader = DataLoader(dataset=tmp_valid_data_set,
                                       batch_size=self.valid_batch_size,
                                       num_workers=4,
                                       pin_memory=torch.cuda.is_available())
        valid_data_loader_original_data = DataLoader(dataset=tmp_valid_data_set_original_data,
                                                     batch_size=self.valid_batch_size,
                                                     num_workers=4,
                                                     pin_memory=torch.cuda.is_available())
        return train_data_loader, valid_data_loader, valid_data_loader_original_data

    def _init_full_data_loaders(self):
        train_data_loader = DataLoader(dataset=self.train_data_set,
                                       sampler=self._get_train_sampler(),
                                       batch_size=self.train_batch_size,
                                       num_workers=4,
                                       pin_memory=torch.cuda.is_available())
        valid_data_loader = DataLoader(dataset=self.valid_data_set,
                                       batch_size=self.valid_batch_size,
                                       num_workers=4,
                                       pin_memory=torch.cuda.is_available())
        valid_data_loader_no_lower_resolution = DataLoader(dataset=self.valid_data_set_original_data,
                                                           batch_size=self.valid_batch_size,
                                                           num_workers=4,
                                                           pin_memory=torch.cuda.is_available())
        return train_data_loader, valid_data_loader, valid_data_loader_no_lower_resolution

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
