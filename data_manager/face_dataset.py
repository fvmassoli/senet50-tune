import os
import sys
import torch
import numpy as np
from PIL import Image
from torch.utils.data import Dataset


pil_alg = {"nearest": Image.NEAREST,  # a nearest-neighbor interpolation (better for quality image)
           "box": Image.BOX,
           "bicubic": Image.BICUBIC,  # 4x4 pixels
           "bilinear": Image.BILINEAR,  # 2x2 pixels
           "lanczos": Image.LANCZOS,  # high quality convolutions-based algorithm with flexible kernel
           "linear": Image.LINEAR,
           "hamming": Image.HAMMING}


class VggFace2Dataset(Dataset):
    def __init__(self, root, resolution, algo_name, lowering_resolution_prob=0, transforms=None):
        self.root = root
        self.classes, self.class_to_idx = self._find_classes()
        self.samples = self._make_dataset()
        self.loader = self._get_loader
        self.algo_name = algo_name
        self.algo = pil_alg[algo_name]
        self.resolution = resolution
        self.transforms = transforms
        self.lowering_resolution_prob = lowering_resolution_prob
        self._create_images_output_dir()
        print("="*81)
        print("="*7, 'Dataset init with: resolution {}, algo {}, lower prob: {}'.format(self.resolution, self.algo, self.lowering_resolution_prob), "="*7)
        print("="*81)

    def _create_images_output_dir(self):
        main_output = './data_manager/output_images'
        if not os.path.isdir(main_output): os.makedirs(main_output)
        sub_dir = os.path.join(main_output, 'res_'+str(self.resolution))
        if not os.path.isdir(sub_dir): os.makedirs(sub_dir)
        out_dir = os.path.join(sub_dir, self.algo_name)
        if not os.path.isdir(out_dir): os.makedirs(out_dir)
        self.out_dir = out_dir

    def _find_classes(self):
        if sys.version_info >= (3, 5):
            classes = [d.name for d in os.scandir(self.root) if d.is_dir()]
        else:
            classes = [d for d in os.listdir(self.root) if os.path.isdir(os.path.join(self.root, d))]
        classes.sort()
        class_to_idx = {classes[i]: i for i in range(len(classes))}
        return classes, class_to_idx

    def _make_dataset(self):
        images = []
        dir = os.path.expanduser(self.root)
        for target in sorted(self.class_to_idx.keys()):
            d = os.path.join(dir, target)
            if not os.path.isdir(d):
                continue
            for root, _, fnames in sorted(os.walk(d)):
                for fname in sorted(fnames):
                    path = os.path.join(root, fname)
                    item = (path, self.class_to_idx[target])
                    images.append(item)
        return images

    def _get_loader(self, path):
        with open(path, 'r') as f:
            img = Image.open(f)
            return img.convert('RGB')

    def _lower_resolution(self, img):
        w_i, h_i = img.size
        r = h_i/float(w_i)
        if h_i<w_i:
            h_n = self.resolution
            w_n = h_n/float(r)
        else:
            w_n = self.resolution
            h_n = w_n*float(r)
        img2 = img.resize((int(w_n), int(h_n)), self.algo)
        img2 = img2.resize((w_i, h_i), self.algo)
        return img2

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, label = self.samples[idx]
        img = self.loader(path)
        if torch.rand(1).item() < self.lowering_resolution_prob:
            img = self._lower_resolution(img)
        if self.transforms:
            img = self.transforms(img)
        return img, label, path
