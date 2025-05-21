import os
import warnings
import torch.utils.data
import torchvision.datasets as datasets
import torchvision.transforms as transforms

import pandas as pd
import time
import numpy as np
import sys
import conf

opt = conf.CIFAR10Opt


class CIFAR10Dataset(torch.utils.data.Dataset):

    def __init__(self, file='../dataset/ichar/minmax_scaling_all.csv',
                 domains=None, activities=None,
                 max_source=100, transform='none'):
        st = time.time()
        self.domains = domains
        self.activity = activities
        self.max_source = max_source

        self.domains = domains

        self.img_shape = opt['img_size']
        self.features = None
        self.class_labels = None
        self.domain_labels = None
        self.file_path = opt['file_path']

        self.sub_path1s = []
        self.sub_path2s = []
        self.data_filenames = []
        self.label_filenames = []
        
        assert (len(domains) > 0)
        for domain in domains:
            if domain.startswith('original'):
                self.sub_path1s += ['origin']
                self.sub_path2s += ['']
                self.data_filenames += ['original.npy']
                self.label_filenames += ['labels.npy']
            elif domain.startswith('test'):
                self.sub_path1s += ['corrupted']
                self.sub_path2s += ['severity-1']  # all data are same in 1~5
                self.data_filenames += ['test.npy']
                self.label_filenames += ['labels.npy']
            elif domain.endswith('-1'):
                self.sub_path1s += ['corrupted']
                self.sub_path2s += ['severity-1']
                self.data_filenames += [domain.split('-')[0] + '.npy']
                self.label_filenames += ['labels.npy']
            elif domain.endswith('-2'):
                self.sub_path1s += ['corrupted']
                self.sub_path2s += ['severity-2']
                self.data_filenames += [domain.split('-')[0] + '.npy']
                self.label_filenames += ['labels.npy']
            elif domain.endswith('-3'):
                self.sub_path1s += ['corrupted']
                self.sub_path2s += ['severity-3']
                self.data_filenames += [domain.split('-')[0] + '.npy']
                self.label_filenames += ['labels.npy']
            elif domain.endswith('-4'):
                self.sub_path1s += ['corrupted']
                self.sub_path2s += ['severity-4']
                self.data_filenames += [domain.split('-')[0] + '.npy']
                self.label_filenames += ['labels.npy']
            elif domain.endswith('-5'):
                self.sub_path1s += ['corrupted']
                self.sub_path2s += ['severity-5']
                self.data_filenames += [domain.split('-')[0] + '.npy']
                self.label_filenames += ['labels.npy']
            elif domain.endswith('_all'):
                self.sub_path1s += ['corrupted']
                self.sub_path2s += ['severity_all']
                self.data_filenames += str(domain[:-4]) + '.npy'
                self.label_filenames += ['labels.npy']

        if transform == 'src':
            self.transform = transforms.Compose(
                [
                    transforms.RandomCrop(32, padding=4),
                    transforms.RandomHorizontalFlip(),
                ])

        elif transform == 'val':
            self.transform = None
        else:
            raise NotImplementedError

        self.preprocessing()

    def preprocessing(self):
        
        self.features = []
        self.class_labels = []
        self.domain_labels = []
        
        for i in range(len(self.sub_path1s)):
            path = f'{self.file_path}/{self.sub_path1s[i]}/{self.sub_path2s[i]}/'

            data = np.load(path + self.data_filenames[i])
            # change NHWC to NCHW format
            data = np.transpose(data, (0, 3, 1, 2))
            # make it compatible with our models (normalize)
            data = data.astype(np.float32)/ 255.0
            self.features += [data]
            self.class_labels += [np.load(path + self.label_filenames[i])]
            # assume that single domain is passed as List
            self.domain_labels += [np.array([i for _ in range(len(data))])]


        self.features = np.concatenate(self.features)
        self.class_labels = np.concatenate(self.class_labels)
        self.domain_labels = np.concatenate(self.domain_labels)
        
        self.dataset = torch.utils.data.TensorDataset(
            torch.from_numpy(self.features),
            torch.from_numpy(self.class_labels),
            torch.from_numpy(self.domain_labels))

    def __len__(self):
        return len(self.dataset)

    def get_num_domains(self):
        return len(self.domains)

    def get_datasets_per_domain(self):
        return self.datasets

    def __getitem__(self, idx):
        if isinstance(idx, torch.Tensor):
            idx = idx.item()
        img, cl, dl = self.dataset[idx]
        if self.transform:
            img = self.transform(img)
        return img, cl, dl


if __name__ == '__main__':
    pass
