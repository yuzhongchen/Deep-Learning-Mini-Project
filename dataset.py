import numpy as np
from torch.utils.data import Dataset
import os
import cv2
import torchvision.transforms as transforms
from PIL import Image


class CIFARDataset(Dataset):
    def __init__(self, data_root='./cifar10', split='train'):
        super(CIFARDataset, self).__init__()

        self.data_root = data_root
        self.split = split
        self.list_path = self.dataset_path(self.data_root, self.split)
        self.images = []
        self.labels = []
        self.mean = np.array([0.4914, 0.4822, 0.4465])
        self.std = np.array([0.2470, 0.2435, 0.2616])
        self.train_transforms = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=self.mean, std=self.std)
        ])
        with open(self.list_path, 'r') as split_list:
            for line in split_list.readlines():
                line = line.split('\n')[0]
                image_path, label_path = line.split('\t')[0], line.split('\t')[1]
                self.images.append(image_path)
                self.labels.append(label_path)

    def __getitem__(self, index):
        img_name = self.images[index]
        img = cv2.imread(img_name.split('\t')[0])
        label = self.labels[index]
        if self.train_transforms is not None:
            img = Image.fromarray(img)
            img = self.train_transforms(img)
        return img, label

    def __len__(self):
        return len(self.images)

    def dataset_path(self, data_root, split):
        split_list = ['train', 'test', 'val']
        if split not in split_list:
            raise NotImplementedError(
                f'Parameter {split} shoule be one of %s ' % split_list
            )
        split_list_path = os.path.join(data_root, '%s.txt' % split)
        return split_list_path
