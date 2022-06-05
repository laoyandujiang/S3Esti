import numpy as np
import cv2
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
import copy
import os
import math
from scipy.ndimage.filters import maximum_filter
from scipy.ndimage.morphology import binary_dilation
from point_select_alg_ring_back import get_circle_win, \
    remove_out_point, get_circle_win_tensor, remove_out_point_with_pos
from show_result_pair_scale_angle import draw_center_line


def get_s_a_error_rela(scale12_pred, angle12_pred, scale12_gt, angle12_gt):
    scale_error_vec = np.abs(1 - scale12_pred / scale12_gt)
    angle12_pred = angle12_pred[:, np.newaxis]
    angle12_gt = angle12_gt[:, np.newaxis]
    angle_error_prob = np.concatenate(
        (angle12_pred - angle12_gt, angle12_pred + math.pi * 2 - angle12_gt,
         angle12_pred - math.pi * 2 - angle12_gt), axis=1)
    angle_error_vec = np.min(np.abs(angle_error_prob), axis=1)
    return scale_error_vec, angle_error_vec


def get_s_a_error(scale1, angle1, scale2, angle2, scale12_gt, angle12_gt):
    scale12_pred = scale2 / (scale1 + 1e-16)
    angle12_pred = (angle2 - angle1) % (math.pi * 2)
    angle12_pred[angle12_pred < -math.pi] += (math.pi * 2)
    angle12_pred[angle12_pred > math.pi] -= (math.pi * 2)

    scale_error_vec, angle_error_vec = get_s_a_error_rela(
        scale12_pred, angle12_pred, scale12_gt, angle12_gt)
    return scale_error_vec, angle_error_vec


class EvaluationSep():
    def __init__(self, point_list: list, desc_list: list, H: np.ndarray,
                 soft_dist_vec, out_dist: int,
                 image_row: int, image_col: int, image_shape_ori: np.ndarray,
                 need_single, pair_txt_path,
                 scale_list=[], angle_list=[],
                 match_image_path: str = None):
        self.each_num = len(point_list)
        self.image_row, self.image_col = image_row, image_col
        self.xy_range = np.array([out_dist, self.image_col - out_dist,
                                  out_dist, self.image_row - out_dist])
        self.need_single = need_single
        self.soft_dist_match = soft_dist_vec
        self.soft_dist_homo = soft_dist_vec
        self.image_shape_ori = image_shape_ori

        self.H = H

        self.point_list = point_list
        self.desc_list = desc_list
        self.scale_list = scale_list
        self.angle_list = angle_list

        self.pair_txt_path = pair_txt_path
        self.match_image_path = match_image_path

        self.try_num = 50
        self.grid_size = 120
        self.large_point_num_ratio = 0.5
        self.need_HAMMING=False

    def map_point_ori_shape(self, point_list: list, image_shape_ori: torch.Tensor):
        each_num = len(point_list)
        point_list_mapped = copy.deepcopy(point_list)
        if image_shape_ori.shape[0] < 1:
            return point_list_mapped

        for image_id in range(each_num):
            shape_ori_now = image_shape_ori[image_id]
            point_ratio = np.array([[shape_ori_now[1] / self.image_col,
                                     shape_ori_now[0] / self.image_row]])
            point_list_mapped[image_id][:, :2] *= point_ratio
        return point_list_mapped

    def grid_num_large_ratio(self, image_row, image_col, point):
        grad_r_num = int(round(image_row / self.grid_size))
        grad_row = int(math.floor(image_row / grad_r_num))
        grad_c_num = int(round(image_col / self.grid_size))
        grad_col = int(math.floor(image_col / grad_c_num))
        grad_id_mat = -np.ones((image_row, image_col), dtype='int')
        row_id_mat = np.arange(grad_r_num).reshape((grad_r_num, 1))
        row_id_mat = np.repeat(row_id_mat, grad_row, axis=0)
        col_id_mat = np.arange(grad_c_num).reshape((1, grad_c_num))
        col_id_mat = np.repeat(col_id_mat, grad_col, axis=1)
        grad_id_mat_exist = row_id_mat * grad_c_num + col_id_mat
        grad_id_mat[:grad_id_mat_exist.shape[0], :grad_id_mat_exist.shape[1]] = grad_id_mat_exist
        point_int = np.floor(point).astype('int')
        point_belong_id = grad_id_mat[point_int[:, 1], point_int[:, 0]]
        grad_num = grad_r_num * grad_c_num
        grad_id_bins = np.arange(grad_num + 1)
        grad_each_num, _ = np.histogram(point_belong_id, bins=grad_id_bins)
        point_num = point.shape[0]
        point_num_thre = point_num / grad_num * self.large_point_num_ratio
        num_large_ratio = np.sum(grad_each_num >= point_num_thre) / grad_num

        return num_large_ratio

    def get_single_point_id(self, point1, feature1, feature2):
        point1 = point1[:, :2].copy()
        point_num = point1.shape[0]
        dist = 2 - 2 * np.clip(np.dot(feature1, feature2.T), -1, 1)
        min_dist = np.min(dist, axis=1)
        point1_r = np.expand_dims(point1, 1)
        point1_c = np.expand_dims(point1, 0)
        dist1_self = np.linalg.norm(point1_r - point1_c, ord=None, axis=2)
        same_pos = (dist1_self < 0.5)
        large_dist = 9999
        same_dist = np.full(same_pos.shape, fill_value=large_dist, dtype=dist.dtype)
        min_dist_rep = np.tile(min_dist[np.newaxis, :], (point_num, 1))
        same_dist[same_pos] = min_dist_rep[same_pos]
        min_dist_id = np.argmin(same_dist, axis=1)
        single_id = np.unique(min_dist_id)
        return single_id

    def get_pair_repeat(self, point1_img, point2_img, shape_ori_1, shape_ori_2, H_1_2):
        point1 = point1_img[:, :2].copy()
        point2 = point2_img[:, :2].copy()
        point1_num = point1.shape[0]
        point2_num = point2.shape[0]
        point_num = (point1_num + point2_num) / 2
        if point1_num < 1 or point2_num < 1:
            return 0, point_num

        point_ratio1 = np.array([[shape_ori_1[1] / self.image_col,
                                  shape_ori_1[0] / self.image_row]])
        point_ratio2 = np.array([[shape_ori_2[1] / self.image_col,
                                  shape_ori_2[0] / self.image_row]])
        # 1 to 2
        point_here = point1 * point_ratio1
        point_here = point_here.astype('float32')[np.newaxis, :]
        point1_warped = cv2.perspectiveTransform(point_here, H_1_2)
        point1_warped = point1_warped.squeeze(0)
        point1_warped /= point_ratio2
        point1_warped = remove_out_point(point1_warped, self.xy_range)
        # 2 to 1
        point_here = point2 * point_ratio2
        point_here = point_here.astype('float32')[np.newaxis, :]
        point2_warped = cv2.perspectiveTransform(point_here, np.linalg.inv(H_1_2))
        point2_warped = point2_warped.squeeze(0)
        point2_warped /= point_ratio1
        point2_warped = remove_out_point(point2_warped, self.xy_range)

        # Compute the repeatability
        N1 = point1_warped.shape[0]
        N2 = point2_warped.shape[0]
        point1 = np.expand_dims(point1, 1)
        point2_warped = np.expand_dims(point2_warped, 0)
        point2 = np.expand_dims(point2, 1)
        point1_warped = np.expand_dims(point1_warped, 0)
        # shapes are broadcasted to N1 x N2 x 2:
        dist1 = np.linalg.norm(point1 - point2_warped,
                               ord=None, axis=2)
        dist2 = np.linalg.norm(point2 - point1_warped,
                               ord=None, axis=2)
        count1 = 0
        count2 = 0
        repeatability = 0
        if N2 != 0:
            min1 = np.min(dist1, axis=1)
            correct_mat = (min1[:, np.newaxis] <= self.soft_dist_match[np.newaxis, :])
            count1 = np.sum(correct_mat.astype('int'), axis=0)
        if N1 != 0:
            min2 = np.min(dist2, axis=1)
            correct_mat = (min2[:, np.newaxis] <= self.soft_dist_match[np.newaxis, :])
            count2 = np.sum(correct_mat.astype('int'), axis=0)
        if N1 + N2 > 0:
            repeatability = (count1 + count2) / (N1 + N2)

        return repeatability, point_num

    def get_corr_point_ind(self, point1_img, point2_img, shape_ori_1, shape_ori_2,
                           H_2_1, image_list, soft_dist, result_index):
        point1 = point1_img[:, :2].copy()
        point2 = point2_img[:, :2].copy()
        point1_num = point1.shape[0]
        point2_num = point2.shape[0]

        point_ratio1 = np.array([[shape_ori_1[1] / self.image_col,
                                  shape_ori_1[0] / self.image_row]])
        point_ratio2 = np.array([[shape_ori_2[1] / self.image_col,
                                  shape_ori_2[0] / self.image_row]])
        # 2 to 1
        point_here = point2 * point_ratio2
        point_here = point_here.astype('float32')[np.newaxis, :]
        point2_warped = cv2.perspectiveTransform(point_here, H_2_1)
        point2_warped = point2_warped.squeeze(0)
        point2_warped /= point_ratio1

        # Compute the repeatability
        point1_expdim = np.expand_dims(point1, 1)
        point2_warped = np.expand_dims(point2_warped, 0)
        # shapes are broadcasted to N1 x N2 x 2:
        dist1 = np.linalg.norm(point1_expdim - point2_warped,
                               ord=None, axis=2)
        min1_ind = np.argmin(dist1, axis=1)
        min_dist = dist1[np.arange(dist1.shape[0]), min1_ind]
        pair_pos1 = (min_dist <= soft_dist)
        pair_ind1 = np.where(pair_pos1)[0]
        pair_ind2 = min1_ind[pair_pos1]

        center2 = point2[pair_ind2]

        # obtain the true relative scale and orientation
        ratio_h1_inv = np.array([[self.image_col / shape_ori_1[1], 0, 0],
                                 [0, self.image_row / shape_ori_1[0], 0],
                                 [0, 0, 1]], dtype='float32')
        ratio_h2 = np.array([[shape_ori_2[1] / self.image_col, 0, 0],
                             [0, shape_ori_2[0] / self.image_row, 0],
                             [0, 0, 1]], dtype='float32')
        H_inv_size = np.dot(ratio_h1_inv, np.dot(H_2_1, ratio_h2))
        H_pair = {(0, 1): H_inv_size}
        scale_dict, angle_dict = draw_center_line(
            'bip_%d' % result_index, image_list, H_pair, center2, 'debug_dir_eval', False)
        scale_rela_gt = scale_dict[(0, 1)]
        angle_rela_gt = angle_dict[(0, 1)]
        angle_rela_gt = angle_rela_gt * math.pi / 180
        return pair_ind1, pair_ind2, scale_rela_gt, angle_rela_gt

    def get_pair_s_a_esti(self, point1_img, point2_img, shape_ori_1, shape_ori_2,
                          feature1, feature2, H_1_2,
                          scale1, scale2, angle1, angle2, image1, image2):
        if self.need_single:
            single_id1 = self.get_single_point_id(point1_img, feature1, feature2)
            single_id2 = self.get_single_point_id(point2_img, feature2, feature1)
            point1_img = point1_img[single_id1]
            scale1 = scale1[single_id1]
            angle1 = angle1[single_id1]
            point2_img = point2_img[single_id2]
            scale2 = scale2[single_id2]
            angle2 = angle2[single_id2]

        soft_dist = 3

        s_error_list1 = []
        a_error_list1 = []
        s_error_list = []
        a_error_list = []
        for k in range(2):
            if k == 0:
                point1_img_val = point1_img
                point2_img_val = point2_img
                scale1_val = scale1
                angle1_val = angle1
                scale2_val = scale2
                angle2_val = angle2
                shape_ori_1_val = shape_ori_1
                shape_ori_2_val = shape_ori_2
                H_2_1 = np.linalg.inv(H_1_2)
                image_list = [image2, image1]
            else:
                point1_img_val = point2_img
                point2_img_val = point1_img
                scale1_val = scale2
                angle1_val = angle2
                scale2_val = scale1
                angle2_val = angle1
                shape_ori_1_val = shape_ori_2
                shape_ori_2_val = shape_ori_1
                H_2_1 = H_1_2
                image_list = [image1, image2]
            pair_ind1, pair_ind2, scale_rela_gt, angle_rela_gt = self.get_corr_point_ind(
                point1_img_val, point2_img_val, shape_ori_1_val, shape_ori_2_val,
                H_2_1, image_list, soft_dist, k)
            scale1_pair = scale1_val[pair_ind1]
            angle1_pair = angle1_val[pair_ind1]
            scale2_pair = scale2_val[pair_ind2]
            angle2_pair = angle2_val[pair_ind2]
            s_error_now, a_error_now = get_s_a_error(
                scale1_pair, angle1_pair, scale2_pair, angle2_pair,
                1 / scale_rela_gt, -angle_rela_gt)
            s_error_list.append(s_error_now)
            a_error_list.append(a_error_now)

            s_error_now, a_error_now = get_s_a_error(
                scale1_pair, angle1_pair, scale2_pair, angle2_pair,
                scale_rela_gt, angle_rela_gt)
            s_error_list1.append(s_error_now)
            a_error_list1.append(a_error_now)

        s_error_vec = np.concatenate(s_error_list, axis=0)
        a_error_vec = np.concatenate(a_error_list, axis=0)
        a_error_vec = a_error_vec * 180 / math.pi
        s_error = np.mean(s_error_vec)
        a_error = np.mean(a_error_vec)
        sa_num = s_error_vec.shape[0]

        s_error_vec1 = np.concatenate(s_error_list1, axis=0)
        a_error_vec1 = np.concatenate(a_error_list1, axis=0)
        a_error_vec1 = a_error_vec1 * 180 / math.pi
        s_error1 = np.mean(s_error_vec1)
        a_error1 = np.mean(a_error_vec1)

        return s_error, a_error, s_error_vec, a_error_vec, sa_num

    def get_pair_match_score(self, point1_img, point2_img, shape_ori_1, shape_ori_2,
                             feature1, feature2, real_H, match_ind, match_ind_inv,
                             point1_matched=None, point2_matched=None):
        if self.need_single:
            single_id1 = self.get_single_point_id(point1_img, feature1, feature2)
            single_id2 = self.get_single_point_id(point2_img, feature2, feature1)
            point1_img = point1_img[single_id1]
            feature1 = feature1[single_id1]
            point2_img = point2_img[single_id2]
            feature2 = feature2[single_id2]
            match_ind = None
            match_ind_inv = None

        point1 = point1_img[:, :2].copy()
        point2 = point2_img[:, :2].copy()
        point1_num = point1.shape[0]
        point2_num = point2.shape[0]
        match_score = 0

        if point1_num < 1 or point2_num < 1:
            return match_score, 0

        point_ratio1 = np.array([[shape_ori_1[1] / self.image_col,
                                  shape_ori_1[0] / self.image_row]])
        point_ratio2 = np.array([[shape_ori_2[1] / self.image_col,
                                  shape_ori_2[0] / self.image_row]])
        point1 = point1 * point_ratio1
        point2 = point2 * point_ratio2

        if (point1_matched is not None) and (point2_matched is not None):
            point1_matched = point1_matched * point_ratio1
            point2_matched = point2_matched * point_ratio2

        if self.need_HAMMING:
            feature1 = feature1.astype(np.uint8)
            feature2 = feature2.astype(np.uint8)
            bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=False)
        else:
            bf = cv2.BFMatcher(cv2.NORM_L2, crossCheck=False)

        score1, true_num1 = self.calcu_score_from_desc_pair(
            feature1, feature2, bf, point1, point2, real_H, point_ratio2, match_ind,
            point1_matched, point2_matched)
        score2, true_num2 = self.calcu_score_from_desc_pair(
            feature2, feature1, bf, point2, point1, np.linalg.inv(real_H),
            point_ratio1, match_ind_inv, point2_matched, point1_matched)

        true_num = (true_num1 + true_num2) / 2
        match_score = np.zeros((score1.shape[0]))
        for k in range(score1.shape[0]):
            if score1[k] < 0 or score2[k] < 0:
                match_score[k] = -1
            else:
                match_score[k] = (score1[k] + score2[k]) / 2

        return match_score, true_num

    def calcu_score_from_desc_pair(self, feature1, feature2, bf_obj, point1, point2,
                                   real_H, point_ratio2, match_ind,
                                   point1_matched=None, point2_matched=None):
        if (point1_matched is not None) and (point2_matched is not None):
            pass
        elif match_ind is None:
            matches = bf_obj.match(feature1, feature2)
            matches_idx = np.array([m.queryIdx for m in matches])
            point1_matched = point1[matches_idx, :]
            matches_idx = np.array([m.trainIdx for m in matches])
            point2_matched = point2[matches_idx, :]
        else:
            point1_matched = point1[match_ind[:, 0], :]
            point2_matched = point2[match_ind[:, 1], :]
        point1_here = point1_matched.astype('float32')[np.newaxis, :]
        point2_true = cv2.perspectiveTransform(point1_here, real_H)
        point2_true = point2_true.squeeze(0)
        point2_matched = point2_matched / point_ratio2
        point2_true = point2_true / point_ratio2
        point2_here = np.c_[point2_true, point2_matched]
        point2_inner = remove_out_point(point2_here, self.xy_range)
        point2_true = point2_inner[:, :2]
        point2_matched = point2_inner[:, 2:]
        dist_vec = np.linalg.norm(point2_matched - point2_true, axis=1)
        correct_mat = (dist_vec[:, np.newaxis] < self.soft_dist_match[np.newaxis, :])
        match_num = np.sum(correct_mat.astype('int'), axis=0)
        point2_true_inner = remove_out_point(point2_true, self.xy_range)
        exist_num = point2_true_inner.shape[0]
        if exist_num == 0:
            return -np.ones((self.soft_dist_match.shape[0])), np.zeros((self.soft_dist_match.shape[0]))
        match_score = match_num / exist_num
        return match_score, match_num

    def get_pair_homography(self, point1_img, point2_img, shape_ori_1, shape_ori_2,
                            feature1, feature2, real_H,
                            write_match_mark=False,
                            image1: np.ndarray = None, image2: np.ndarray = None,
                            match_cross_ind=None, point1_matched=None, point2_matched=None):
        if self.need_single:
            single_id1 = self.get_single_point_id(point1_img, feature1, feature2)
            single_id2 = self.get_single_point_id(point2_img, feature2, feature1)
            point1_img = point1_img[single_id1]
            feature1 = feature1[single_id1]
            point2_img = point2_img[single_id2]
            feature2 = feature2[single_id2]
            match_cross_ind = None

        point1 = point1_img[:, :2].copy()
        point2 = point2_img[:, :2].copy()

        point1_num = point1.shape[0]
        point2_num = point2.shape[0]

        draw_radius = 3
        draw_margin = 10
        white_image = image1[:, :draw_margin, :].copy()
        white_image[:] = 255

        if write_match_mark:
            show_image1 = image1.copy()
            show_image2 = image2.copy()
            point1_int = np.round(point1).astype('int')
            point2_int = np.round(point2).astype('int')
            for p_now in point1_int:
                show_image1 = cv2.circle(show_image1, (p_now[0], p_now[1]), radius=draw_radius,
                                         color=(0, 0, 255), thickness=-1)
            for p_now in point2_int:
                show_image2 = cv2.circle(show_image2, (p_now[0], p_now[1]), radius=draw_radius,
                                         color=(0, 0, 255), thickness=-1)
            image_write = np.concatenate((show_image1, white_image, show_image2), axis=1)
        else:
            show_image1 = None
            show_image2 = None
            image_write = None

        point_ratio1 = np.array([[shape_ori_1[1] / self.image_col,
                                  shape_ori_1[0] / self.image_row]])
        point_ratio2 = np.array([[shape_ori_2[1] / self.image_col,
                                  shape_ori_2[0] / self.image_row]])
        point1 = point1 * point_ratio1
        point2 = point2 * point_ratio2

        if (point1_matched is not None) and (point2_matched is not None):
            point1_matched = point1_matched * point_ratio1
            point2_matched = point2_matched * point_ratio2

        num_ratio1 = self.grid_num_large_ratio(shape_ori_1[0], shape_ori_1[1], point1)
        num_ratio2 = self.grid_num_large_ratio(shape_ori_2[0], shape_ori_2[1], point2)
        num_ratio_detect = (num_ratio1 + num_ratio2) / 2

        soft_thre_num = self.soft_dist_homo.shape[0]
        if point1_num < 1 or point2_num < 1:
            correct_mean = np.zeros((soft_thre_num,))
            correct_std = np.zeros((soft_thre_num,))
            return {'correctness': correct_mean,
                    'homo_dist': -1,
                    'correctness_std': correct_std,
                    'homo_dist_std': 0,
                    'homography': None,
                    'point1_matched': 0,
                    'point2_matched': 0,
                    'inlier': None,
                    'image_write': image_write,
                    'num_ratio_detect': num_ratio_detect,
                    'num_ratio_match': 0,
                    'num_ratio_ransac': 0}

        # Match the keypoints with the warped_keypoints with nearest neighbor search
        if self.need_HAMMING:
            feature1 = feature1.astype(np.uint8)
            feature2 = feature2.astype(np.uint8)
            bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
        else:
            bf = cv2.BFMatcher(cv2.NORM_L2, crossCheck=True)

        if (point1_matched is not None) and (point2_matched is not None):
            pass
        elif match_cross_ind is None:
            matches = bf.match(feature1, feature2)
            if len(matches) < 1:
                bf = cv2.BFMatcher(cv2.NORM_L2, crossCheck=False)
                matches = bf.match(feature1, feature2)
            matches_idx = np.array([m.queryIdx for m in matches])
            point1_matched = point1[matches_idx, :]
            matches_idx = np.array([m.trainIdx for m in matches])
            point2_matched = point2[matches_idx, :]
        else:
            point1_matched = point1[match_cross_ind[:, 0], :]
            point2_matched = point2[match_cross_ind[:, 1], :]
        point_match_num = point1_matched.shape[0]

        if point_match_num > 0:
            num_ratio_match1 = self.grid_num_large_ratio(shape_ori_1[0], shape_ori_1[1], point1_matched)
            num_ratio_match2 = self.grid_num_large_ratio(shape_ori_2[0], shape_ori_2[1], point2_matched)
            num_ratio_match = (num_ratio_match1 + num_ratio_match2) / 2
        else:
            num_ratio_match = 0

        # Estimate the homography between the matches using RANSAC
        try_num = self.try_num
        mean_dist_vec = np.zeros((try_num,), dtype='float')
        correctness_vec = np.zeros((try_num, soft_thre_num), dtype='float')
        H = None
        inliers = np.zeros((point1_matched.shape[0])) > 1
        inliers_num = 0
        num_ratio_ransac = 0
        for try_id in range(try_num):
            random_pos = np.random.permutation(point_match_num)
            point1_matched_now = point1_matched[random_pos, :]
            point2_matched_now = point2_matched[random_pos, :]

            H_now, inliers_now = cv2.findHomography(point1_matched_now,
                                                    point2_matched_now, cv2.RANSAC)
            inliers_now = inliers_now.flatten().astype('bool')

            if H_now is None:
                mean_dist_vec[try_id] = 100
                correctness_vec[try_id] = 0
                continue
            inliers_now_temp = np.zeros((point1_matched.shape[0])) > 1
            inliers_now_temp[random_pos] = inliers_now
            inliers_now = inliers_now_temp

            point1_inlier = point1_matched[inliers_now, :]
            point2_inlier = point2_matched[inliers_now, :]
            if point1_inlier.size > 0:
                num_ratio_ransac1 = self.grid_num_large_ratio(shape_ori_1[0], shape_ori_1[1], point1_inlier)
                num_ratio_ransac2 = self.grid_num_large_ratio(shape_ori_2[0], shape_ori_2[1], point2_inlier)
                num_ratio_ransac = (num_ratio_ransac1 + num_ratio_ransac2) / 2
            else:
                num_ratio_ransac = 0

            inliers_num_now = sum(inliers_now)
            if inliers_num_now > inliers_num:
                H = H_now
                inliers = inliers_now
                inliers_num = inliers_num_now

            corners = np.array([[0, 0, 1],
                                [0, shape_ori_1[0] - 1, 1],
                                [shape_ori_1[1] - 1, 0, 1],
                                [shape_ori_1[1] - 1, shape_ori_1[0] - 1, 1]])
            real_warped_corners = np.dot(corners, np.transpose(real_H))
            real_warped_corners = real_warped_corners[:, :2] / real_warped_corners[:, 2:]
            real_warped_corners /= point_ratio1
            warped_corners = np.dot(corners, np.transpose(H_now))
            warped_corners = warped_corners[:, :2] / warped_corners[:, 2:]
            warped_corners /= point_ratio1
            mean_dist = np.mean(np.linalg.norm(real_warped_corners - warped_corners, axis=1))
            correctness = (mean_dist <= self.soft_dist_homo).astype('float32')

            mean_dist_vec[try_id] = mean_dist
            correctness_vec[try_id] = correctness

        if write_match_mark:
            point1_homo = point1_matched[inliers, :] / point_ratio1
            point2_homo = point2_matched[inliers, :] / point_ratio2
            point1_homo = np.round(point1_homo).astype('int')
            point2_homo = np.round(point2_homo).astype('int')
            for p_now in point1_homo:
                show_image1 = cv2.circle(show_image1, (p_now[0], p_now[1]), radius=draw_radius,
                                         color=(255, 0, 0), thickness=-1)
            for p_now in point2_homo:
                show_image2 = cv2.circle(show_image2, (p_now[0], p_now[1]), radius=draw_radius,
                                         color=(255, 0, 0), thickness=-1)
            image_write = np.concatenate((show_image1, white_image, show_image2), axis=1)
            for line_id in range(point1_homo.shape[0]):
                p1 = (point1_homo[line_id, 0], point1_homo[line_id, 1])
                p2 = (point2_homo[line_id, 0] + show_image1.shape[1] + draw_margin, point2_homo[line_id, 1])
                show_image2 = cv2.line(image_write, p1, p2, color=(0, 255, 0), thickness=1)

        if H is None:
            correct_mean = np.zeros((soft_thre_num,))
            correct_std = np.zeros((soft_thre_num,))
            return {'correctness': correct_mean,
                    'homo_dist': -1,
                    'correctness_std': correct_std,
                    'homo_dist_std': 0,
                    'homography': H,
                    'point1_matched': point1_matched / point_ratio1,
                    'point2_matched': point2_matched / point_ratio2,
                    'inlier': inliers,
                    'image_write': image_write,
                    'num_ratio_detect': num_ratio_detect,
                    'num_ratio_match': num_ratio_match,
                    'num_ratio_ransac': num_ratio_ransac}

        correct_mean = np.mean(correctness_vec, axis=0)
        correct_std = np.std(correctness_vec, axis=0)
        dist_mean, dist_std = np.mean(mean_dist_vec), np.std(mean_dist_vec)

        return {'correctness': correct_mean,
                'homo_dist': dist_mean,
                'correctness_std': correct_std,
                'homo_dist_std': dist_std,
                'homography': H,
                'point1_matched': point1_matched / point_ratio1,
                'point2_matched': point2_matched / point_ratio2,
                'inlier': inliers,
                'image_write': image_write,
                'num_ratio_detect': num_ratio_detect,
                'num_ratio_match': num_ratio_match,
                'num_ratio_ransac': num_ratio_ransac}

    def get_homograghy_esti(self, write_match_mark=False,
                            image_name=None, image_ori: np.ndarray = None):
        assert (self.each_num >= 2)
        assert len(self.desc_list) == self.each_num
        if self.match_image_path is None:
            write_match_mark = False
        soft_dist_num = self.soft_dist_homo.shape[0]
        homo_corr_sum = np.zeros(soft_dist_num)
        match_score_sum = np.zeros(soft_dist_num)
        true_num_sum = np.zeros(soft_dist_num)
        repeat_sum = np.zeros(soft_dist_num)
        scale_error_list = []
        angle_error_list = []
        point_num_sum = 0
        homo_corr_std_sum = 0
        num_ratio_detect_sum = 0
        num_ratio_match_sum = 0
        num_ratio_ransac_sum = 0
        pair_num = 0
        for id_1 in range(self.each_num):
            for id_2 in range(id_1 + 1, self.each_num):
                H_1_ref = np.linalg.inv(self.H[id_1])
                H_ref_2 = self.H[id_2]
                H_1_2 = np.dot(H_ref_2, H_1_ref)
                shape_ori_1 = self.image_shape_ori[id_1]
                shape_ori_2 = self.image_shape_ori[id_2]
                repeat_now, point_num_now = self.get_pair_repeat(self.point_list[id_1],
                                                                 self.point_list[id_2],
                                                                 shape_ori_1, shape_ori_2, H_1_2)
                s_error, a_error, sa_num = -1, -1, 0
                if len(self.scale_list) > 0:
                    s_error, a_error, s_error_vec, a_error_vec, sa_num = self.get_pair_s_a_esti(
                        self.point_list[id_1], self.point_list[id_2],
                        shape_ori_1, shape_ori_2,
                        self.desc_list[id_1], self.desc_list[id_2], H_1_2,
                        self.scale_list[id_1], self.scale_list[id_2],
                        self.angle_list[id_1], self.angle_list[id_2],
                        image_ori[id_1], image_ori[id_2])

                match_ind, match_ind_inv, match_cross_ind = None, None, None

                match_score_now, true_num_now = self.get_pair_match_score(
                    self.point_list[id_1], self.point_list[id_2],
                    shape_ori_1, shape_ori_2,
                    self.desc_list[id_1], self.desc_list[id_2], H_1_2,
                    match_ind, match_ind_inv)
                homo_result = self.get_pair_homography(self.point_list[id_1], self.point_list[id_2],
                                                       shape_ori_1, shape_ori_2,
                                                       self.desc_list[id_1], self.desc_list[id_2],
                                                       H_1_2,
                                                       write_match_mark,
                                                       image_ori[id_1], image_ori[id_2],
                                                       match_cross_ind)
                repeat_str = ','.join(('%.4f' % item) for item in repeat_now)
                mscore_str = ','.join(('%.4f' % item) for item in match_score_now)
                homo_str = ','.join(('%.3f' % item) for item in homo_result['correctness'])
                homo_std_str = ','.join(('%.3f' % item) for item in homo_result['correctness_std'])
                sa_str = ('%.3f,%.3f' % (s_error, a_error))
                str = ('%s-%d_%d,HA:,%s,HA_std:,%s,MS:,%s,Rep:,%s,SA_error,%s,%.3f,%.3f' %
                       (image_name, id_1, id_2, homo_str, homo_std_str,
                        mscore_str, repeat_str, sa_str,
                        homo_result['homo_dist'],
                        homo_result['homo_dist_std']))
                if sa_num > 0:
                    scale_item_str = ','.join(('%.3f' % item) for item in s_error_vec)
                    angle_item_str = ','.join(('%.3f' % item) for item in a_error_vec)
                    str = '%s,s_each:,%s,a_each:,%s' % (str, scale_item_str, angle_item_str)
                str += '\n'
                with open(self.pair_txt_path, 'a') as f:
                    f.write(str)
                if write_match_mark:
                    path_now = os.path.join(self.match_image_path, image_name)
                    if not os.path.exists(path_now):
                        os.mkdir(path_now)
                    image_write = homo_result['image_write']
                    image_write_name = os.path.join(
                        path_now, '%d_%d-corr_%s-dist_%.3f_%.4f-score_%s-rep_%s.png' %
                                  (id_1, id_2,
                                   homo_str,
                                   homo_result['homo_dist'],
                                   homo_result['homo_dist_std'],
                                   mscore_str, repeat_str))
                    plt.imsave(image_write_name, image_write)

                if np.all(match_score_now >= 0):
                    homo_corr_sum += homo_result['correctness']
                    match_score_sum += match_score_now
                    true_num_sum += true_num_now
                    repeat_sum += repeat_now
                    point_num_sum += point_num_now
                    homo_corr_std_sum += homo_result['correctness_std']
                    num_ratio_detect_sum += homo_result['num_ratio_detect']
                    num_ratio_match_sum += homo_result['num_ratio_match']
                    num_ratio_ransac_sum += homo_result['num_ratio_ransac']
                    pair_num += 1
                if sa_num > 0:
                    scale_error_list.append(s_error_vec)
                    angle_error_list.append(a_error_vec)

        if pair_num < 1:
            return np.zeros((5), dtype='float32')
        homo_corr = homo_corr_sum / pair_num
        match_score = match_score_sum / pair_num
        repeat = repeat_sum / pair_num
        true_num = true_num_sum / pair_num
        homo_corr_std = homo_corr_std_sum / pair_num
        point_num = point_num_sum / pair_num
        num_ratio_detect = num_ratio_detect_sum / pair_num
        num_ratio_match = num_ratio_match_sum / pair_num
        num_ratio_ransac = num_ratio_ransac_sum / pair_num
        scale_error_mean, scale_error_std, angle_error_mean, angle_error_std = -1, -1, -1, -1
        if len(scale_error_list) > 0:
            scale_error_vec = np.concatenate(scale_error_list, axis=0)
            scale_error_mean = np.mean(scale_error_vec)
            scale_error_std = np.std(scale_error_vec)
            angle_error_vec = np.concatenate(angle_error_list, axis=0)
            angle_error_mean = np.mean(angle_error_vec)
            angle_error_std = np.std(angle_error_vec)

        result_dict = {'repeat': repeat, 'homo_corr': homo_corr, 'm_score': match_score,
                       'homo_corr_std': homo_corr_std, 'point_num': point_num,
                       'true_num': true_num, 'num_ratio_detect': num_ratio_detect,
                       'num_ratio_match': num_ratio_match, 'num_ratio_ransac': num_ratio_ransac,
                       'scale_error': scale_error_mean, 'scale_error_std': scale_error_std,
                       'angle_error': angle_error_mean, 'angle_error_std': angle_error_std}

        return result_dict
