import os
import torch
import numpy as np
import matplotlib.pyplot as plt
import cv2
import random
import glob
from torch.utils.data import Dataset

class ToTensorHPatches(object):
    def __call__(self, sample):
        image = sample['image']
        image = image.transpose([0, 3, 1, 2])
        image_gray = sample['image_gray']
        sample_tensor = {'image': (torch.from_numpy(image).float() - 128) / 129,
                         'H': sample['H'],
                         'image_shape': sample['image_shape'],
                         'image_name': sample['image_name'],
                         'image_gray': torch.from_numpy(image_gray).float() / 255,
                         'image_ori': sample['image_ori'],
                         'image_name_list': sample['image_name_list']}
        return sample_tensor


class HPatchesDataset(Dataset):
    def __init__(self, root_dir: str, suffix: str,
                 image_row: int, image_col: int, type_str: str):
        self.root_dir = root_dir
        self.suffix = suffix
        self.image_row = image_row
        self.image_col = image_col
        self.type_str = type_str

        if self.type_str == 'v':
            self.dir_names = glob.glob(os.path.join(self.root_dir, 'v_*'))
        elif self.type_str == 'i':
            self.dir_names = glob.glob(os.path.join(self.root_dir, 'i_*'))
        else:
            self.dir_names = glob.glob(os.path.join(self.root_dir, '*'))

        self.dir_names = sorted(self.dir_names)

        self.to_tensor_fun = ToTensorHPatches()

    def __len__(self):
        return len(self.dir_names)

    def __getitem__(self, item):
        dir_name = self.dir_names[item]
        image_names = glob.glob(os.path.join(dir_name, '*' + self.suffix))
        image_num = len(image_names)
        assert image_num > 1

        image_num = min(20, image_num)

        image_name_list = []
        base_number = 1
        image_name_now = os.path.join(dir_name, '%d%s' % (base_number, self.suffix))
        image_base = cv2.imread(image_name_now)
        image_base = cv2.cvtColor(image_base, cv2.COLOR_BGR2RGB)
        image_base_shape = np.array(image_base.shape)
        H_base = np.eye(3, dtype='float32')
        image_base = cv2.resize(image_base, (self.image_col, self.image_row))
        image_base_gray = cv2.cvtColor(image_base, cv2.COLOR_RGB2GRAY)
        image_name_list.append(os.path.basename(image_name_now))

        image_array = np.tile(image_base, [image_num, 1, 1, 1])
        H = np.tile(H_base, [image_num, 1, 1])
        image_shape = np.tile(image_base_shape, [image_num, 1])
        image_gray_array = np.tile(image_base_gray, [image_num, 1, 1, 1])
        image_ori = np.tile(image_base, [image_num, 1, 1, 1])

        for image_id in range(1, image_num):
            image_name_now = os.path.join(dir_name, '%d%s' % (image_id + 1, self.suffix))
            image_now = cv2.imread(image_name_now)
            image_now = cv2.cvtColor(image_now, cv2.COLOR_BGR2RGB)
            image_shape[image_id] = np.array(image_now.shape)
            image_array[image_id] = cv2.resize(image_now, (self.image_col, self.image_row))
            H_name_now = 'H_%d_%d' % (base_number, image_id + 1)
            H_fullname_now = os.path.join(dir_name, H_name_now)
            if os.path.exists(H_fullname_now):
                H[image_id] = np.loadtxt(H_fullname_now)
            else:
                H[image_id] = np.eye(3, dtype='float32')
            image_gray_array[image_id][0] = cv2.cvtColor(image_array[image_id], cv2.COLOR_RGB2GRAY)
            image_ori[image_id] = image_array[image_id].copy()

            image_name_list.append(os.path.basename(image_name_now))

        sample = {'image': image_array, 'H': H, 'image_shape': image_shape,
                  'image_name': os.path.basename(dir_name),
                  'image_gray': image_gray_array,
                  'image_ori': image_ori,
                  'image_name_list': image_name_list}

        sample = self.to_tensor_fun(sample)

        return sample
