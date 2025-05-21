from torchvision.datasets import VisionDataset
import warnings
import torch
from PIL import Image
import os
import os.path
import numpy as np
from torchvision import transforms


def pil_loader(path):
    # open path as file to avoid ResourceWarning (https://github.com/python-pillow/Pillow/issues/835)
    with open(path, 'rb') as f:
        img = Image.open(f)
        return img.convert('RGB')


class ImageList(VisionDataset):
    """
    Args:
        root (string): Root directory of dataset
        transform (callable, optional): A function/transform that  takes in an PIL image
            and returns a transformed version. E.g, ``transforms.RandomCrop``
        target_transform (callable, optional): A function/transform that takes in the
            target and transforms it.
    """

    def __init__(self, feats=None, labels=None, transform=None, target_transform=None,  empty=False):
        super(ImageList, self).__init__(root=None, transform=transform, target_transform=target_transform)

        self.empty = empty
        if empty:
            self.samples = np.empty((1, 2), dtype='<U1000')
            # self.samples = None
        else:
            self.samples = list(zip(feats, labels))
            # self.samples = dict(zip(range(len(data)), data))
            pass

    def __getitem__(self, index):
        if len(self.samples[index]) == 3:
            feat, label, binary = self.samples[index]
        else:
            feat, label = self.samples[index]
            binary = -1
        label = int(label)

        output = {
            'label': label,
            'path': "",
            'index': index
        }

        output['img'] = feat
        output['img0'] = feat
        output['binary'] = binary

        return output

    def __len__(self):
        return len(self.samples)

    def add_item(self, addition):
        if self.empty:
            self.samples = addition
            self.empty = False
        else:
            self.samples += addition
            # self.samples = np.concatenate((self.samples, addition), axis=0)
        return self.samples

    def remove_item(self, reduced):
        for index in reduced:
            self.samples.pop(index)
        # self.samples = np.delete(self.samples, reduced, axis=0)
        return self.samples
