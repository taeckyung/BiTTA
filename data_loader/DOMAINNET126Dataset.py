import os
from PIL import Image
import torch
import torchvision.transforms as transforms
import conf

def load_image(img_path):
    img = Image.open(img_path)
    img = img.convert("RGB")
    return img

opt = conf.DOMAINNET126Opt


class DOMAINNET126Dataset(torch.utils.data.Dataset):
    def __init__(self, file='',
                 domains=None, activities=None,
                 max_source=100, transform='none'
    ):
        # self.image_root = image_root
        # self._label_file = label_file
        # self.transform = transform
        
        # self.file_path = opt['file_path']

        # assert (
        #     label_file or pseudo_item_list
        # ), f"Must provide either label file or pseudo labels."
        
        self.domains = domains
        self.image_root = f"dataset/domainnet-126"
        label_file = f"dataset/domainnet-126/{domains[0]}_list.txt"

        self.item_list = (
            self.build_index(label_file)
        )
        
        if transform == 'src':
            self.transform =  transforms.Compose(
            [
                transforms.Resize((256, 256)),
                transforms.RandomCrop(224),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
            ]
        )

        elif transform == 'val':
            self.transform = transforms.Compose(
            [
                transforms.Resize((256, 256)),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
            ]
        )
        else:
            raise NotImplementedError
        
    def get_num_domains(self):
        return 1
    
    def build_index(self, label_file):
        """Build a list of <image path, class label> items.

        Args:
            label_file: path to the domain-net label file

        Returns:
            item_list: a list of <image path, class label> items.
        """
        # read in items; each item takes one line
        with open(label_file, "r") as fd:
            lines = fd.readlines()
        lines = [line.strip() for line in lines if line]

        item_list = []
        for item in lines:
            img_file, label = item.split()
            img_path = os.path.join(self.image_root, img_file)
            label = int(label)
            item_list.append((img_path, label, img_file))

        return item_list

    def __getitem__(self, idx):
        """Retrieve data for one item.

        Args:
            idx: index of the dataset item.
        Returns:
            img: <C, H, W> tensor of an image
            label: int or <C, > tensor, the corresponding class label. when using raw label
                file return int, when using pseudo label list return <C, > tensor.
        """
        img_path, label, _ = self.item_list[idx]
        img = load_image(img_path)
        if self.transform:
            img = self.transform(img)

        return img, label, idx

    def __len__(self):
        return len(self.item_list)
