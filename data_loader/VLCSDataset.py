
import torch.utils.data
import torchvision.datasets as datasets
import torchvision.transforms as transforms

import time
import conf
from PIL import ImageFile


opt = conf.VLCSOpt

ImageFile.LOAD_TRUNCATED_IMAGES = True


class VlcsDataset(torch.utils.data.Dataset):

    def __init__(self, file='',
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

        assert (len(domains) > 0)
        # assert (domains[0] in opt['domains'] + ['test'])
        assert (set(domains).issubset(set(opt['domains'] + ['test'])))

        if domains[0] == "test":
            self.domains[0] = opt['src_domains'][0]
            
        self.sub_paths = self.domains      
        
        if transform == 'src':
            self.transform = transforms.Compose([
            transforms.RandomResizedCrop(224, scale=(0.7, 1.0)),
            transforms.RandomHorizontalFlip(),
            transforms.ColorJitter(0.3, 0.3, 0.3, 0.3),
            transforms.RandomGrayscale(),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
        
        elif transform == 'val':
            self.transform = transforms.Compose([
            transforms.Resize((224,224)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
            
        else:
            raise NotImplementedError

        self.preprocessing()

    def preprocessing(self):

        self.dataset = []
        self.domain_list = []
        for sub_path_i, sub_path in enumerate(self.sub_paths):
            path = f'{self.file_path}/{sub_path}/'
            dataset = datasets.ImageFolder(path, transform=self.transform)
            self.dataset += [dataset]
            self.domain_list += ([sub_path_i] * len(dataset))
        self.dataset = torch.utils.data.ConcatDataset(self.dataset)


    def __len__(self):
        return len(self.dataset)

    def get_num_domains(self):
        return len(self.domains)

    def get_datasets_per_domain(self):
        raise NotImplementedError
        # return self.dataset

    def __getitem__(self, idx):
        if isinstance(idx, torch.Tensor):
            idx = idx.item()
        img, cl = self.dataset[idx]
        dl = self.domain_list[idx]
        
        cl = torch.tensor([cl])
        dl = torch.tensor([dl])
        
        return img, cl, dl


if __name__ == '__main__':
    pass
