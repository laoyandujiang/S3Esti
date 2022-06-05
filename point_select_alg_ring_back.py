import numpy as np
import cv2
import torch
import matplotlib.pyplot as plt
from scipy.ndimage.filters import maximum_filter
from scipy.ndimage.morphology import binary_dilation

def get_circle_win(rad: int, dtype: str):
    win_size = 2 * rad + 1
    win = np.zeros((win_size, win_size), dtype=dtype)
    c_mat = np.tile(np.arange(-rad, rad + 1, dtype='int'), (win_size, 1))
    r_mat = c_mat.transpose((1, 0))
    win[(r_mat * r_mat + c_mat * c_mat) <= rad * rad] = 1
    return win


def get_circle_win_tensor(rad: int, dtype: str):
    win_size = 2 * rad + 1
    win = torch.zeros((win_size, win_size), dtype=dtype)
    c_mat = np.tile(np.arange(-rad, rad + 1, dtype='int'), (win_size, 1))
    r_mat = c_mat.transpose((1, 0))
    fore_pos = ((r_mat * r_mat + c_mat * c_mat) <= rad * rad)
    win.view(-1)[np.where(fore_pos.reshape(-1))] = 1
    return win


def get_rand_ring(out_rad: int, in_rad: int, rand_num: int, dtype: str):
    assert (out_rad > in_rad)
    out_circle = get_circle_win(out_rad, dtype='int')
    in_circle = np.zeros_like(out_circle)
    rad_diff = out_rad - in_rad
    in_circle[rad_diff:-rad_diff, rad_diff:-rad_diff] = get_circle_win(in_rad, dtype='int')
    ring = out_circle - in_circle
    ring_pos = np.where(ring.reshape(-1) > 0)[0]
    max_num = len(ring_pos)
    if max_num <= rand_num:
        return ring.astype(dtype)
    rand_ring = np.zeros_like(ring).astype(dtype)
    rand_pos = np.random.choice(ring_pos, rand_num, replace=False)
    rand_ring.put(rand_pos, 1)
    return rand_ring


def remove_out_point(point: np.ndarray, xy_range: np.ndarray):
    if point.size < 1:
        return point
    point_result = point[(point[:, 0] > xy_range[0]) &
                         (point[:, 0] < xy_range[1]) &
                         (point[:, 1] > xy_range[2]) &
                         (point[:, 1] < xy_range[3]), :]
    return point_result


def remove_out_point_with_pos(point: np.ndarray, xy_range: np.ndarray):
    if point.size < 1:
        return point, np.ones((point.shape[0],), dtype='bool')
    pos = ((point[:, 0] > xy_range[0]) & (point[:, 0] < xy_range[1]) &
           (point[:, 1] > xy_range[2]) & (point[:, 1] < xy_range[3]))
    point_result = point[pos, :]
    return point_result, pos


def get_print_dict(fore_num_cum, fore_loss_cum, back_num_cum,
                   back_loss_cum, repeat_thre_cum, image_base_num,
                   image_real_num):
    return {'fore_num_cum': fore_num_cum,
            'fore_loss_cum': fore_loss_cum,
            'back_num_cum': back_num_cum,
            'back_loss_cum': back_loss_cum,
            'repeat_thre_cum': repeat_thre_cum,
            'image_base_num': image_base_num,
            'image_real_num': image_real_num}
