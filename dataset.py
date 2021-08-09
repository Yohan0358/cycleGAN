import os
import glob
import numpy as np

import torch
from torch.utils.data import Dataset, DataLoader

import torchvision.transforms as transforms

from PIL import Image

from utils import *

class Custom_dataset(Dataset):
    def __init__(self, img_path, transforms, mode = 'train'):
        super(Custom_dataset, self).__init__()
        
        if mode == 'train':
            self.path_monet = img_path[0]
            self.path_photo = img_path[1]
        if mode == 'test':
            self.path_photo = img_path[1]
        
        self.mode = mode
        self.transforms = transforms
        
    def __getitem__(self, idx):
         
        if self.mode == 'train':
            monet_img = self.path_monet[idx]
            monet_img = Image.open(monet_img).convert('RGB')
            monet_img = self.transforms(monet_img)

            
            photo_idx = int(np.random.randint(0, len(self.path_photo), (1,)))
            photo_img = self.path_photo[photo_idx]
            photo_img = Image.open(photo_img).convert('RGB')
            photo_img = self.transforms(photo_img)

            return monet_img, photo_img

        elif self.mode == 'test':
            photo_img = self.path_photo[idx]
            photo_img = Image.open(photo_img).convert('RGB')
            photo_img = self.transforms(photo_img)
            return photo_img
        
    def __len__(self):
        if self.mode == 'train':
            return len(self.path_monet)

        elif self.mode == 'test':
            return len(self.path_photo)

if __name__ == '__main__':
    
    img_path = 'cycle_GAN/dataset/'
    monet_path = glob.glob(img_path + 'monet_jpg/*')
    photo_path = glob.glob(img_path + 'photo_jpg/*')

    transform = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize([0.5, 0.5, 0.5], 
                                    [0.5, 0.5, 0.5])
    ])

    test = Custom_dataset([monet_path, photo_path], transforms=transform, mode = 'train')
    test_loader = DataLoader(test, 1, shuffle=True)

    a, b = next(iter(test_loader))
    print(a.shape, b.shape)
