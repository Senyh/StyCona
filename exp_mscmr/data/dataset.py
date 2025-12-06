import sys
import os
import random
import torch
from torch.utils.data.dataset import Dataset
from exp_mscmr.data import transforms as T
from torchvision.transforms import *
import torchvision.transforms.functional as TF
import numpy as np
import h5py
from PIL import Image
from copy import deepcopy
from wheels.stycona_utils import StyConTransform


class MSCMRDataset(Dataset):
    def __init__(self, image_path='', image_size=256, stage='train', modality='C0', is_augmentation=False, labeled=False, percentage=0.1, stycona=False):
        super(MSCMRDataset, self).__init__()
        self.image_size = image_size
        self.sep = '\\' if sys.platform[:3] == 'win' else '/'
        self.stage = stage
        self.is_augmentation = is_augmentation
        self.image_path = image_path
        self.labeled = labeled
        self.stycona = stycona
        
        if self.stage == 'train':
            with open(self.image_path + "/train36.list", "r") as f1:
                patient_list = f1.readlines()
            patient_list = [item.replace("\n", "") for item in patient_list]
            if labeled:
                patient_list = patient_list[:int(len(patient_list)*percentage)]
            else:
                patient_list = patient_list[int(len(patient_list)*percentage):]
            train_slices_list = [item for item in os.listdir(os.path.join(self.image_path, 'MS_CMR_OriSize_h5py')) if item.find(modality) != -1]
            self.sample_list = [x for y in patient_list for x in train_slices_list if x.startswith(y + '_')]
        else:
            with open(self.image_path + "/val9.list", "r") as f1:
                patient_list = f1.readlines()
            patient_list = [item.replace("\n", "") for item in patient_list]
            val_slices_list = [item for item in os.listdir(os.path.join(self.image_path, 'MS_CMR_OriSize_h5py')) if item.find(modality) != -1]
            self.sample_list = [x for y in patient_list for x in val_slices_list if x.startswith(y + '_')]
        if self.is_augmentation:
            self.augmentation = self.augmentation_transform()
        self.post_transform = self.post_transform()
        self.label_transform = self.label_transform()
        self.pre_transform = self.pre_transform()
        if stycona: self.stycon_transform = self.stycon_transform()

    def __getitem__(self, item):
        if self.stage == 'train':
            case = self.sample_list[item]
            h5f = h5py.File(os.path.join(self.image_path, 'MS_CMR_OriSize_h5py', case), "r")
            image = h5f["image"][:] * 255.
            label = h5f["label"][:]
            image = Image.fromarray(image).convert('L')
            label = Image.fromarray(label).convert('L')
            image, label = self.pre_transform(image, label)
            imageA1, imageA2 = deepcopy(image), deepcopy(image)
            imageA1, _ = self.augmentation(imageA1, label)
            imageA2, _ = self.augmentation(imageA1, label)
            image, label = self.post_transform(image), self.label_transform(label)
            imageA1 = self.post_transform(imageA1)
            imageA2 = self.post_transform(imageA2)
            if self.stycona:
                if torch.rand(1) < 0.5:
                    acase = random.choice(self.sample_list)
                    ah5f = h5py.File(self.image_path + "/MS_CMR_OriSize_h5py/{}".format(acase), "r")
                    aimage = ah5f["image"][:] * 255.
                    alabel = ah5f["label"][:]
                    aimage = Image.fromarray(aimage).convert('L')
                    alabel = Image.fromarray(alabel).convert('L')
                    aimage, alabel = self.pre_transform(aimage, alabel)
                    aimageA1, _ = self.augmentation(aimage, label)
                    aimageA2, _ = self.augmentation(aimage, label)
                    aimageA1 = self.post_transform(aimageA1)
                    aimageA2 = self.post_transform(aimageA2)
                    imageA1 = self.stycon_transform(imageA1, aimageA1)
                    imageA2 = self.stycon_transform(imageA2, aimageA2)
            label = torch.from_numpy(np.array(label)).unsqueeze(0).float()
            return image, label, imageA1, imageA2
        else:
            case = self.sample_list[item]
            h5f = h5py.File(os.path.join(self.image_path, 'MS_CMR_OriSize_h5py', case), "r")
            image = h5f["image"][:] * 255.
            label = h5f["label"][:]
            image = Image.fromarray(image).convert('L')
            label = Image.fromarray(label).convert('L')
            image, label = self.post_transform(image), self.label_transform(label)
            label = torch.from_numpy(np.array(label)).unsqueeze(0).float()
            return image, label

    def __len__(self):
        return len(self.sample_list)

    @staticmethod
    def augmentation_transform():
        return T.Compose([
            T.ColorJitter(0.5, 0.5, 0.5, 0.25),
            T.RandomAutocontrast(p=0.2),
            T.RandomEqualize(p=0.2),
            T.GaussianBlur(kernel_size=3, sigma=(0.1, 2.0)),
        ])
    
    def pre_transform(self):
        return T.Compose([
            T.RandomHorizontalFlip(),
            T.RandomVerticalFlip(),
            T.RandomRotation(degrees=180),
        ])

    def post_transform(self):
        return Compose([
            Resize([self.image_size, self.image_size], InterpolationMode.BILINEAR),
            ToTensor(),
        ])

    def label_transform(self):
        return Compose([
            Resize([self.image_size, self.image_size], InterpolationMode.NEAREST)
        ])
    
    def stycon_transform(self):
        return StyConTransform()

