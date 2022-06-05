import os
import glob
import cv2
import numpy as np
from torch.utils.data import Dataset


class HPatchesHomoDataset(Dataset):
    def __init__(self, root_dir, image_ext, image_row, image_col):
        base_number = 1
        H_base = np.eye(3, dtype='float32')
        self.root_dir = root_dir
        self.image_ext = image_ext
        self.image_row = image_row
        self.image_col = image_col
        self.dir_names = os.listdir(root_dir)
        self.dir_names = sorted(self.dir_names)
        dir_num = len(self.dir_names)
        self.H_dict = {}
        self.H_pair_dict = {}
        for dir_id in range(dir_num):
            dir_name_now = self.dir_names[dir_id]
            full_dir_name = os.path.join(root_dir, dir_name_now)
            image_names = glob.glob(os.path.join(full_dir_name, '*' + image_ext))
            image_names = sorted(image_names)
            image_num = len(image_names)
            H = np.tile(H_base, [image_num, 1, 1])
            image1 = cv2.imread(image_names[0])
            x_ratio = image_col / image1.shape[1]
            y_ratio = image_row / image1.shape[0]
            resize_homo1 = np.array([[x_ratio, 0, 0], [0, y_ratio, 0], [0, 0, 1]],
                                    dtype='float32')
            for image_id in range(1, image_num):
                H_name_now = 'H_%d_%d' % (base_number, image_id + 1)
                H_fullname_now = os.path.join(full_dir_name, H_name_now)
                if os.path.exists(H_fullname_now):
                    h_now = np.loadtxt(H_fullname_now)
                else:
                    h_now = np.eye(3, dtype='float32')
                image2 = cv2.imread(image_names[image_id])
                x_ratio = image_col / image2.shape[1]
                y_ratio = image_row / image2.shape[0]
                resize_homo2 = np.array([[x_ratio, 0, 0], [0, y_ratio, 0], [0, 0, 1]],
                                        dtype='float32')
                H[image_id] = np.dot(resize_homo2,
                                     np.dot(h_now, np.linalg.inv(resize_homo1)))
            self.H_dict[dir_name_now] = H
            each_num = H.shape[0]
            H_pair_now = {}
            for id_1 in range(each_num):
                for id_2 in range(id_1 + 1, each_num):
                    H_1_ref = np.linalg.inv(H[id_1])
                    H_ref_2 = H[id_2]
                    H_1_2 = np.dot(H_ref_2, H_1_ref)
                    H_pair_now[(id_1, id_2)] = H_1_2
            self.H_pair_dict[dir_name_now] = H_pair_now

        self.name_list = list(self.H_dict.keys())

    def __len__(self):
        return len(self.name_list)

    def __getitem__(self, item):
        dir_name_now = self.dir_names[item]
        full_dir_name = os.path.join(self.root_dir, dir_name_now)
        image_names = glob.glob(os.path.join(full_dir_name, '*' + self.image_ext))
        image_names = sorted(image_names)
        image_num = len(image_names)
        assert image_num > 1

        image_list = []
        for image_id in range(0, image_num):
            image_name_now = os.path.join(full_dir_name,
                                          '%d%s' % (image_id + 1, self.image_ext))
            image_now = cv2.imread(image_name_now)
            image_now = cv2.resize(image_now, (self.image_col, self.image_row))
            image_list.append(image_now)

        return image_list
