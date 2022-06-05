import os
import torch
import cv2
import numpy as np
import matplotlib.pyplot as plt
import torch.nn.functional as F
import random
import imgaug as ia
import math
import copy
from torch.utils.data import Dataset
from imgaug import augmenters as iaa

avail_image_ext = ['.jpg', '.jpeg', '.tif', '.tiff', '.png',
                   '.bmp', '.gif', '.ppm']


def get_rand_vec(limit, patch_num):
    # generate the value randomly
    rand_vec = (limit[0] + (limit[1] - limit[0]) * np.random.random((patch_num,)))
    return rand_vec


def get_random_border_point(border_x1_ratio, point_num):
    border_x2_ratio = 1 - border_x1_ratio
    x_ratio = np.random.random((point_num,))
    x_inner_pos = ((x_ratio > border_x1_ratio) & (x_ratio < border_x2_ratio))
    x_out_pos = (~x_inner_pos)
    x_out_num = np.sum(x_out_pos)
    # re-sample the points if the original points are close to the image border
    min_border_ratio = 0.2
    if x_out_num > 0:
        # randomly sample the points in the left or right border
        side2_pos = (np.random.random((x_out_num,)) > 0.5)
        among_border_ratio = ((1 - min_border_ratio) * np.random.random((x_out_num,)) +
                              min_border_ratio)
        new_x_ratio = among_border_ratio * border_x1_ratio[x_out_pos]
        new_x_ratio[side2_pos] = 1 - new_x_ratio[side2_pos]
        x_ratio[x_out_pos] = new_x_ratio
    return x_ratio


def get_affine_patch(tensor, centers, patch_size, scale_vec, angle_vec,
                     scale_ratio_list, is_train):
    # use different "align_corners" parameters according to the version of pytorch
    torch_version = float('.'.join((torch.__version__).split('.')[:2]))
    torch_version_thre = 1.2

    patch_num = centers.shape[0]
    patch_size_half = round(patch_size / 2)
    corner = np.array([[-patch_size_half, -patch_size_half],
                       [-patch_size_half, patch_size_half],
                       [patch_size_half, -patch_size_half],
                       [patch_size_half, patch_size_half]]).astype('float32')
    # generate the scales and orientations randomly
    scale_vec = scale_vec.cpu().numpy() if torch.is_tensor(scale_vec) else scale_vec
    angle_vec = angle_vec.cpu().numpy() if torch.is_tensor(angle_vec) else angle_vec
    centers = centers.cpu().numpy() if torch.is_tensor(centers) else centers
    sin_angle = np.sin(angle_vec)
    cos_angle = np.cos(angle_vec)
    # scale_ratio_list: the ratios used to generate the multiple-scale input patches
    patch_list = []
    for k, scale_ratio_now in enumerate(scale_ratio_list):
        # during the training, randomly resize the image and then crop the patches
        if is_train:
            pre_ratio_l = [0.8, 1.5]
        else:
            # during the testing, directly crop the patches on the original image
            pre_ratio_l = [1, 1]
        rand_pre_ratio = pre_ratio_l[0] + (pre_ratio_l[1] - pre_ratio_l[0]) * random.random()
        tensor_now = F.interpolate(tensor, scale_factor=rand_pre_ratio,
                                   mode='bilinear', align_corners=True)
        scale_vec_now = scale_vec * rand_pre_ratio
        centers_now = centers * rand_pre_ratio
        channel_num, row, col = tensor_now.shape[1:4]

        mat_list = [get_trans_mat(sin_angle[pos], cos_angle[pos], scale_vec_now[pos] * scale_ratio_now)
                    for pos in range(patch_num)]
        trans_corner = [cv2.perspectiveTransform(corner[np.newaxis, :], H_mat)
                        for H_mat in mat_list]
        trans_corner = [trans_corner[pos].squeeze(0) + centers_now[pos:pos + 1] for
                        pos in range(patch_num)]
        # get the transformation parameters
        # the coordinates should be mapped to [-1,1] to satisfy the affine_grid process
        corner_norm = corner / patch_size_half
        trans_corner_norm = [get_norm_xy(item, row, col) for item in trans_corner]
        theta = [cv2.getPerspectiveTransform(corner_norm, trans_corner_norm[pos])[np.newaxis, :]
                 for pos in range(patch_num)]
        theta = np.concatenate(theta, axis=0)
        theta = torch.from_numpy(theta[:, :2, :].astype('float32'))
        # get the transformed patches
        grid_size = torch.Size((theta.shape[0], channel_num, patch_size, patch_size))
        if torch_version > torch_version_thre:
            grid = F.affine_grid(theta, grid_size, align_corners=True)
        else:
            grid = F.affine_grid(theta, grid_size)
        grid = grid.view(1, grid.shape[0], patch_size * patch_size, 2)
        grid = grid.to(tensor_now.device)
        if torch_version > torch_version_thre:
            patch_now = F.grid_sample(tensor_now, grid, padding_mode='zeros',
                                      mode='nearest', align_corners=True)
        else:
            patch_now = F.grid_sample(tensor_now, grid, padding_mode='zeros', mode='nearest')
        patch_now = patch_now.view(tensor_now.shape[1], patch_num, patch_size, patch_size)
        patch_now = patch_now.transpose(0, 1)
        patch_list.append(patch_now)

    patch = torch.cat(patch_list, dim=1)
    return patch


def get_norm_xy(xy, row, col):
    xy_new = np.c_[(xy[:, 0] / (col / 2.)) - 1., (xy[:, 1] / (row / 2.)) - 1.]
    return xy_new


def get_trans_mat(sin_v, cos_v, scale):
    mat = np.array([[scale * cos_v, -scale * sin_v, 0],
                    [scale * sin_v, scale * cos_v, 0],
                    [0, 0, 1]], dtype='float32')
    return mat


def get_affine_tensor_batch(scale, angle):
    num = scale.shape[0]
    sin_v = torch.sin(angle)
    cos_v = torch.cos(angle)
    tensor = torch.zeros((num, 2, 3), device=scale.device, dtype=scale.dtype)
    tensor[:, 0, 0] = scale * cos_v
    tensor[:, 0, 1] = -scale * sin_v
    tensor[:, 1, 0] = scale * sin_v
    tensor[:, 1, 1] = scale * cos_v
    return tensor


def remove_out_point(point, xy_range):
    # xy_range:x1,x2,y1,y2
    if point.size < 1:
        return point
    inner_pos = ((point[:, 0] > xy_range[0]) & (point[:, 0] < xy_range[1]) &
                 (point[:, 1] > xy_range[2]) & (point[:, 1] < xy_range[3]))
    point_result = point[inner_pos, :]
    return point_result, inner_pos


def to_tensor(image_np):
    tensor = torch.from_numpy(image_np.transpose((0, 3, 1, 2))).float()
    new_limit = [-1, 1]
    tensor = new_limit[0] + (tensor / 255) * (new_limit[1] - new_limit[0])
    return tensor


def load_file_current_dir(dir_path):
    all_list = os.listdir(dir_path)
    all_list = sorted(all_list)
    all_list = [os.path.join(dir_path, name_now) for name_now in all_list]
    file_list = []
    for name_now in all_list:
        if os.path.isfile(name_now):
            if os.path.splitext(name_now)[-1] in avail_image_ext:
                file_list.append(name_now)
            else:
                continue
        else:
            sub_file_list = load_file_current_dir(name_now)
            file_list.extend(sub_file_list)

    return file_list


class ImageSet(Dataset):
    """Face Landmarks dataset."""

    def __init__(self, image_set_path, patch_size, patch_num, scale_limit, angle_limit,
                 is_train, flatten_dir=False, scale_ratio_list=None):
        self.patch_size = patch_size
        self.patch_num = patch_num
        self.angle_limit = copy.deepcopy(angle_limit)
        self.is_train = is_train
        self.ill_seq = self.get_ill_seq()
        self.channel_num = 3
        # the input of model is the multiple-scale patch
        if scale_ratio_list is None:
            self.scale_ratio_list = [0.5, 1, 2]
        else:
            self.scale_ratio_list = scale_ratio_list

        # the range of the scale value
        self.scale_limit = scale_limit

        if not flatten_dir:
            names = os.listdir(image_set_path)
            names = [name_now for name_now in names if os.path.splitext(name_now)[1] in avail_image_ext]
            names = sorted(names)
            self.names = [os.path.join(image_set_path, name_now) for name_now in names]
        else:
            self.names = load_file_current_dir(image_set_path)
            self.names = sorted(self.names)

    def __len__(self):
        return len(self.names)

    def __getitem__(self, idx):
        name = self.names[idx]
        image_ori = cv2.imread(name)
        if len(image_ori.shape) == 2:
            image_ori = np.tile(image_ori[:, :, np.newaxis], (1, 1, 3))
        image_ori = cv2.cvtColor(image_ori, cv2.COLOR_BGR2RGB)
        row, col = image_ori.shape[0], image_ori.shape[1]
        image_show = image_ori.copy()

        # randomly flip the patch
        image = self.flip_transpose_image(image_ori)
        images = np.tile(image[np.newaxis, :], [2, 1, 1, 1])
        # during the training, argument the training samples with the
        # illumination changes to improve the generalization ability
        if self.is_train:
            # different images will be transformed with different illumination changes
            images = self.random_ill_change(images)

        # randomly generate the scale and angle parameters
        tensors = to_tensor(images)
        tensor0 = tensors[0:1]
        tensor1 = tensors[1:2]
        # the angle parameters are sampled uniformly
        angle0 = get_rand_vec(self.angle_limit, self.patch_num)
        angle1 = get_rand_vec(self.angle_limit, self.patch_num)
        # randomly sample the scale of the first image
        scale0 = get_rand_vec([1, self.scale_limit[1]], self.patch_num)
        # randomly sample the scale change, and then obtain the scale of the second image
        scale1_rela_ratio = get_rand_vec([1, self.scale_limit[1]], self.patch_num)
        scale1 = scale0 * scale1_rela_ratio
        # randomly swap the items in scale0 and scale1 to improve the generalization
        exchange_pos = (np.random.random((self.patch_num,)) < 0.5)
        scale0_rem = scale0[exchange_pos]
        scale0[exchange_pos] = scale1[exchange_pos]
        scale1[exchange_pos] = scale0_rem

        # sample the centroid point of the patches
        # the sample border is determined by the scale range
        # obtain the x coordinate
        dist_vec = scale0 * (self.patch_size / 2)
        border_x1_ratio = dist_vec / col
        x_ratio = get_random_border_point(border_x1_ratio, self.patch_num)
        x_vec = x_ratio * col
        # obtain the y coordinate
        border_y1_ratio = dist_vec / row
        y_ratio = get_random_border_point(border_y1_ratio, self.patch_num)
        y_vec = y_ratio * row
        centers = np.c_[x_vec, y_vec].astype('float32')

        # extract the patches according to the scale and angle parameters
        patch0 = get_affine_patch(
            tensor0, centers, self.patch_size, scale0, angle0,
            self.scale_ratio_list, self.is_train)
        patch1 = get_affine_patch(
            tensor1, centers, self.patch_size, scale1, angle1,
            self.scale_ratio_list, self.is_train)

        # # the sampled patches can be saved
        # show_path_now = 'debug_img'
        # max_show_num = 100
        # patch0_show = ((patch0.cpu().numpy().transpose((0, 2, 3, 1)) + 1) * 127).astype('uint8')
        # patch1_show = ((patch1.cpu().numpy().transpose((0, 2, 3, 1)) + 1) * 127).astype('uint8')
        # for pos in range(min(patch0.shape[0], max_show_num)):
        #     p0 = cv2.cvtColor(patch0_show[pos, :, :, 3:6], cv2.COLOR_BGR2RGB)
        #     p1 = cv2.cvtColor(patch1_show[pos, :, :, 3:6], cv2.COLOR_BGR2RGB)
        #     # p0 = cv2.cvtColor(patch0_show[pos, :, :, 6:], cv2.COLOR_BGR2RGB)
        #     # p1 = cv2.cvtColor(patch1_show[pos, :, :, 6:], cv2.COLOR_BGR2RGB)
        #     cv2.imwrite(os.path.join(show_path_now, '%d_1.png' % pos), p0)
        #     cv2.imwrite(os.path.join(show_path_now, '%d_2.png' % pos), p1)

        # get the relative scales and angles
        scale0_to1 = torch.from_numpy(1 / (scale1 / scale0))
        angle0_to1 = torch.from_numpy(angle1 - angle0)
        angle0_to1 = angle0_to1 % (math.pi * 2)
        # guarantee that the angle difference are in [-pi,pi]
        angle0_to1[angle0_to1 < -math.pi] += (math.pi * 2)
        angle0_to1[angle0_to1 > math.pi] -= (math.pi * 2)

        if self.is_train:
            sample = {'patch1': patch0, 'patch2': patch1,
                      'scale': scale0_to1, 'angle': angle0_to1,
                      'center': centers, 'image_name': name}
        else:
            sample = {'patch1': patch0, 'patch2': patch1,
                      'scale': scale0_to1, 'angle': angle0_to1,
                      'full_image': tensors, 'center': centers,
                      'image_name': name}
        return sample

    def random_trans_img(self, image_ori):
        image_num, row, col = image_ori.shape[:3]
        range_ratio = 1 / 8
        range_row, range_col = round(row * range_ratio), round(col * range_ratio)
        # get the random homography matrices, and apply them on the argument images
        point_src = np.array([[1, 1], [col, 1], [1, row], [col, row]], dtype='float32')
        while True:
            point_dst = point_src.copy()
            point_dst += np.c_[np.random.randint(-range_col, range_col, 4),
                               np.random.randint(-range_row, range_row, 4)].astype('float32')

            h_now = cv2.getPerspectiveTransform(point_src, point_dst)
            if abs(np.linalg.det(h_now)) > 1e-5:
                break

        image_new = np.zeros([image_num, row, col, 3], dtype=image_ori.dtype)
        H = h_now.astype('float32')
        for image_id in range(image_num):
            image_new[image_id] = cv2.warpPerspective(
                image_ori[image_id], H, (col, row))

        return image_new, H

    def random_ill_change(self, image_ori):
        image_new = self.ill_seq.augment_images(image_ori)

        image_num = image_new.shape[0]
        shadow_num = round(image_num / 3)
        for image_id in tuple(np.random.choice(np.arange(image_num),
                                               shadow_num, replace=False)):
            image_new[image_id] = self.insert_shadow(image_new[image_id])
        return image_new

    def flip_transpose_image(self, image_ori):
        row, col = image_ori.shape[:2]
        rand_value = random.random()
        if rand_value < 0.6:
            rand_code = 1
        elif rand_value < 0.8:
            rand_code = 0
        else:
            rand_code = -1
        image_new = cv2.flip(image_ori, flipCode=rand_code)
        return image_new

    def get_ill_seq(self):
        light_change = 50
        seq = iaa.Sequential([
            iaa.Sometimes(0.5, iaa.OneOf([
                iaa.WithColorspace(
                    to_colorspace="HSV",
                    from_colorspace="RGB",
                    children=iaa.OneOf([
                        iaa.WithChannels(0, iaa.Add((-5, 5))),
                        iaa.WithChannels(1, iaa.Add((-20, 20))),
                        iaa.WithChannels(2, iaa.Add((-light_change, light_change))),
                    ])
                ),
                iaa.Grayscale((0.2, 0.6)),
                iaa.ChannelShuffle(1),
                iaa.Add((-light_change, light_change)),
                iaa.Multiply((0.5, 1.5)),
            ])),

            iaa.Sometimes(0.5, iaa.OneOf([
                iaa.ContrastNormalization((0.5, 1.5)),
            ])),

            iaa.Sometimes(0.5, iaa.OneOf([
                iaa.AdditiveGaussianNoise(0, (3, 6)),
                iaa.AdditivePoissonNoise((3, 6)),
                iaa.JpegCompression((30, 60)),
                iaa.GaussianBlur(sigma=1),
                iaa.AverageBlur((1, 3)),
                iaa.MedianBlur((1, 3)),
            ])),
        ])
        return seq

    def insert_shadow(self, image_ori):
        img_new = image_ori.copy()
        image_row, image_col = image_ori.shape[:2]
        shadow_num = random.randint(5, 50)
        min_size, max_size = 10, round(min(image_ori.shape[:2]) / 4)
        mask = np.zeros(image_ori.shape[:2], np.uint8)
        rect_shrink_ratio = 1 / 3
        ellipse_prob = 0.3
        transparency_range = [0.8, 1]

        for i in range(shadow_num):
            # size of the shadow
            ax = random.randint(min_size, max_size)
            ay = random.randint(min_size, max_size)
            max_rad = max(ax, ay)
            # centroid of the shadow
            x = np.random.randint(max_rad, image_col - max_rad)
            y = np.random.randint(max_rad, image_row - max_rad)
            # shape type of the shadow
            if random.random() < ellipse_prob:
                angle = np.random.rand() * 90
                cv2.ellipse(mask, (x, y), (ax, ay), angle, 0, 360, 255, -1)
            else:
                shr_x_range = round(rect_shrink_ratio * ax)
                shr_y_range = round(rect_shrink_ratio * ay)
                rad_x, rad_y = round(ax / 2), round(ay / 2)
                rect_point = np.array([[x - rad_x, y - rad_y], [x + rad_x, y - rad_y],
                                       [x + rad_x, y + rad_y], [x - rad_x, y + rad_y]], dtype='int32')
                rect_point += np.c_[np.random.randint(-shr_x_range, shr_x_range, 4),
                                    np.random.randint(-shr_y_range, shr_y_range, 4)]
                cv2.fillConvexPoly(mask, rect_point, 255)

        mask = mask > 1
        mask = np.tile(np.expand_dims(mask, axis=2), (1, 1, 3))
        transparency = np.random.uniform(*transparency_range)
        shadow_value = random.choice((0, 255))
        img_new[mask] = img_new[mask] * transparency + shadow_value * (1 - transparency)

        return img_new
