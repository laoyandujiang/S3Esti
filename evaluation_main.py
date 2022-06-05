import numpy as np
import cv2
import torch
import math
import torch.nn.functional as F
import matplotlib.pyplot as plt
import copy
import time
import os
import patch_desc_utils as p_desc
from scipy.ndimage.filters import maximum_filter
from scipy.ndimage.morphology import binary_dilation
from point_select_alg_ring_back import get_circle_win, remove_out_point, get_circle_win_tensor
from torch.utils.data import DataLoader
from dataset_class import HPatchesDataset
from evaluation_class import EvaluationSep
from show_result_pair_scale_angle import draw_center_line

from AffNet.architectures import AffNetFast
from AffNet.HardNet import HardNet, HardTFeatNet
from AffNet.SparseImgRepresenter import ScaleSpaceAffinePatchExtractor
from AffNet.LAF import convertLAFs_to_A23format, LAF2center, LAF2mainSA
from POP.desc_rep_net import DescRepNet
from abso_esti_model.abso_esti_net import EstiNet
from abso_esti_model.abso_esti_utils import get_wrap_patches, get_wrap_patches_multi
from abso_esti_model.abso_esti_image_set import get_affine_patch

class EvaluationMain():
    def __init__(self, device_str='cuda:0'):
        self.methed_name = ''
        self.root_dir = ''
        self.suffix = ''
        self.nms_rad = -1
        self.max_point_num = -1
        self.soft_dist = -1
        self.out_dist = -1
        self.image_row, self.image_col = -1, -1
        self.xy_range = np.array([-1, -2, -1, -2])
        self.eval_match = True
        self.device = torch.device(device_str)
        self.pair_txt_path = None
        self.match_image_path = None
        self.esti_scale_ratio_list = [0.5, 1, 2]

        self.need_single_point_alg = ('AffNet_esti_HardNet', 'POP_esti')

    def set_dataset(self, root_dir: str, suffix: str, pair_txt_path,
                    match_image_path: str = None):
        self.root_dir = root_dir
        self.suffix = suffix
        self.pair_txt_path = pair_txt_path
        self.match_image_path = match_image_path
        if self.match_image_path is not None:
            if not os.path.exists(match_image_path):
                os.makedirs(match_image_path)

    def set_hyper(self, nms_rad: int, soft_dist_vec, out_dist: int,
                  max_point_num: int, image_row: int, image_col: int):
        self.nms_rad = nms_rad
        self.max_point_num = max_point_num
        self.soft_dist_vec = soft_dist_vec
        self.out_dist = out_dist
        self.image_row, self.image_col = image_row, image_col
        self.xy_range = np.array([out_dist, self.image_col - out_dist,
                                  out_dist, self.image_row - out_dist])

    def main(self, methed_name: str, para_dict: dict = None):
        self.methed_name = methed_name
        device = self.device
        if 'resp_thre' in para_dict.keys():
            self.resp_thre = para_dict['resp_thre']
        else:
            self.resp_thre = 0

        large_point_num = para_dict['large_point_num'] if \
            ('large_point_num' in para_dict.keys()) else self.max_point_num

        data_HPatches = HPatchesDataset(self.root_dir, self.suffix,
                                        self.image_row, self.image_col, 'total')
        dataloader = DataLoader(data_HPatches, batch_size=1, shuffle=False)

        if self.methed_name in ['AffNet_esti_HardNet', 'POP_esti']:
            patch_size = 32
            scale_num = 300
            angle_num = 360
            esti_checkpoint_path = 'abso_esti_model/S3Esti_ep30.pth'

            model_scale = EstiNet(need_bn=True, device=device, out_channels=scale_num,
                                  patch_size=patch_size, scale_ratio=self.esti_scale_ratio_list)
            model_angle = EstiNet(need_bn=True, device=device, out_channels=angle_num,
                                  patch_size=patch_size, scale_ratio=self.esti_scale_ratio_list)
            checkpoint = torch.load(esti_checkpoint_path, map_location=device)
            model_scale.load_state_dict(checkpoint['model_scale'], strict=False)
            model_scale.eval()

            model_angle.load_state_dict(checkpoint['model_angle'], strict=False)
            model_angle.eval()

            model_scale.to(device)
            model_angle.to(device)
            temp = 1


        if self.methed_name in ['AffNet_HardNet', 'AffNet_esti_HardNet']:
            AffNet_path = 'AffNet'
            detector_path = os.path.join(AffNet_path, 'AffNet_det.pth')
            PS = 32
            detector_ = AffNetFast(PS=PS)
            checkpoint = torch.load(detector_path)
            start_epoch = checkpoint['epoch']
            detector_.load_state_dict(checkpoint['state_dict'], strict=True)

            torch.cuda.empty_cache()
            # switch to evaluate mode
            detector_.eval()
            AffNet_net = ScaleSpaceAffinePatchExtractor(
                mrSize=5.192, num_features=self.max_point_num, border=5,
                num_Baum_iters=1, AffNet=detector_)
            AffNet_net = AffNet_net.to(device)

        if self.methed_name in ['AffNet_HardNet', 'AffNet_esti_HardNet']:
            AffNet_path = 'AffNet'
            HardNet_path = os.path.join(AffNet_path, 'HardNet++.pth')
            # HardNet
            HardNet_net = HardNet()
            hncheckpoint = torch.load(HardNet_path)
            HardNet_net.load_state_dict(hncheckpoint['state_dict'], strict=True)
            HardNet_net.eval()
            HardNet_net = HardNet_net.to(device)
        elif self.methed_name in ['POP', 'POP_esti']:
            easy_net = DescRepNet()
            checkpoint = torch.load('POP/POP_net_pretrained.pth', map_location=device)
            easy_net.load_state_dict(checkpoint['model_state_dict'], strict=False)
            easy_net.eval()
            easy_net.to(device)
            if self.methed_name == 'POP_rela':
                desc_model = easy_net

        def get_AffNet_HardNet_point(input_, image_ori_):
            image_num_ = image_ori_.shape[0]
            point_list_res = []
            desc_list_res = []
            scale_list_res = []
            angle_list_res = []

            for image_id_ in range(image_num_):
                image_ori_now = image_ori_[image_id_]
                img_mean = np.mean(image_ori_now, axis=2)
                var_image = torch.from_numpy(img_mean.astype(np.float32))
                var_image_reshape = var_image.view(1, 1, var_image.size(0), var_image.size(1))
                var_image_reshape = var_image_reshape.to(device)

                scale_list_now = []
                angle_list_now = []
                with torch.no_grad():
                    time_last = time.time()
                    LAFs, resp = AffNet_net(var_image_reshape)
                    patches = AffNet_net.extract_patches_from_pyr(LAFs, PS=32)
                    desc_now = HardNet_net(patches)

                work_LAFs = convertLAFs_to_A23format(LAFs.cpu().numpy())
                point_num_now = len(work_LAFs)
                point_now = np.zeros((point_num_now, 3), dtype='float32')
                resp = resp.cpu().numpy()
                resp /= np.max(resp)
                point_now[:, 2] = resp
                for i in range(point_num_now):
                    point_now[i, 0:2] = LAF2center(work_LAFs[i, :, :])[0]
                point_list_res.append(point_now)
                desc_now = desc_now.cpu().numpy()
                desc_list_res.append(desc_now)

                scale_now = np.zeros((point_num_now,))
                angle_now = np.zeros((point_num_now,))
                for i in range(point_num_now):
                    scale_local, angle_local = LAF2mainSA(work_LAFs[i, :, :])
                    scale_now[i] = scale_local
                    angle_now[i] = angle_local

                scale_list_res.append(scale_now)
                angle_list_res.append(angle_now)

            return point_list_res, desc_list_res, scale_list_res, angle_list_res

        def get_AffNet_esti_HardNet_point(input_, image_ori_):
            image_num_ = image_ori_.shape[0]
            point_list_res = []
            desc_list_res = []

            for image_id_ in range(image_num_):
                image_ori_now = image_ori_[image_id_]
                img_mean = np.mean(image_ori_now, axis=2)
                var_image = torch.from_numpy(img_mean.astype(np.float32))
                var_image_reshape = var_image.view(1, 1, var_image.size(0), var_image.size(1))
                var_image_reshape = var_image_reshape.to(device)

                with torch.no_grad():
                    time_last = time.time()
                    LAFs, resp = AffNet_net(var_image_reshape)
                    patches = AffNet_net.extract_patches_from_pyr(LAFs, PS=32)
                    desc_now = HardNet_net(patches)

                work_LAFs = convertLAFs_to_A23format(LAFs.cpu().numpy())
                point_num_now = len(work_LAFs)
                point_now = np.zeros((point_num_now, 3), dtype='float32')
                resp = resp.cpu().numpy()
                resp /= np.max(resp)
                point_now[:, 2] = resp
                for i in range(point_num_now):
                    point_now[i, 0:2] = LAF2center(work_LAFs[i, :, :])[0]
                point_list_res.append(point_now)
                desc_now = desc_now.cpu().numpy()
                desc_list_res.append(desc_now)

            scale_list_res = []
            angle_list_res = []
            input_ = input_.to(device)
            patch_desc_size = 32
            time_last = time.time()
            for image_id_ in range(image_num_):
                point_vec = point_list_res[image_id_]
                image_tensor = input_[image_id_:image_id_ + 1]
                scale_angle_now = []
                patch_new, point_vec_new = get_wrap_patches_multi(
                    image_tensor, model_scale, model_angle, point_vec,
                    patch_desc_size, self.esti_scale_ratio_list,
                    scale_angle_result_list=scale_angle_now)
                scale_base = np.ones((point_vec.shape[0],))
                angle_base = np.zeros((point_vec.shape[0],))
                centers = point_vec[:, :2].astype('float32')
                patch_base = get_affine_patch(
                    image_tensor, centers, patch_desc_size, scale_base, angle_base, [1])
                patch_base = (patch_base + 1) * 127
                patch_new = torch.cat([patch_base, patch_new], dim=0)
                patch_new = torch.mean(patch_new, dim=1, keepdim=True)

                patch_num = patch_new.shape[0]
                each_num = min(patch_num, 2000)
                split_num = math.ceil(patch_num / each_num)
                desc_new_list = []
                for split_id in range(split_num):
                    patch_pos = slice(split_id * each_num, min((split_id + 1) * each_num, patch_num), 1)
                    patch_new_now = patch_new[patch_pos]
                    with torch.no_grad():
                        desc_new_local = HardNet_net(patch_new_now)
                    desc_new_local = desc_new_local.cpu().numpy()
                    desc_new_list.append(desc_new_local)
                desc_new_now = np.concatenate(desc_new_list, axis=0)

                desc_list_res[image_id_] = desc_new_now
                point_list_res[image_id_] = np.r_[point_list_res[image_id_], point_vec_new]
                scale_list_res.append(np.r_[scale_base, scale_angle_now[0]])
                angle_list_res.append(np.r_[angle_base, scale_angle_now[1]])

            return point_list_res, desc_list_res, scale_list_res, angle_list_res

        def get_POP_point(input_):
            input_ = input_.to(device)
            with torch.no_grad():
                time_last = time.time()
                score_output, desc = easy_net(input_)
                score = torch.sigmoid(score_output).detach()
                desc = desc.detach()

            time_last = time.time()
            point_list_res = self.get_point_from_resp(score)
            desc_list_res = self.get_easy_desc_from_map(point_list_res, desc)

            scale_list_res = []
            angle_list_res = []
            image_num_ = input_.shape[0]
            for image_id_ in range(image_num_):
                scale_base = np.ones((point_list_res[image_id_].shape[0],))
                angle_base = np.zeros((point_list_res[image_id_].shape[0],))
                scale_list_res.append(scale_base)
                angle_list_res.append(angle_base)

            return point_list_res, desc_list_res, scale_list_res, angle_list_res

        def get_POP_esti_point(input_):
            input_ = input_.to(device)
            with torch.no_grad():
                time_last = time.time()
                score_output, desc = easy_net(input_)
                score = torch.sigmoid(score_output).detach()

            point_list_res = self.get_point_from_resp(score)
            desc_list_res = self.get_easy_desc_from_map(point_list_res, desc)

            image_num_ = input_.shape[0]
            patch_desc_size = 64
            scale_list_res = []
            angle_list_res = []
            for image_id_ in range(image_num_):
                point_vec = point_list_res[image_id_]
                scale_base = np.ones((point_vec.shape[0],))
                angle_base = np.zeros((point_vec.shape[0],))
                image_tensor = input_[image_id_:image_id_ + 1]
                scale_angle_now = []
                patch_new, point_vec_new = get_wrap_patches_multi(
                    image_tensor, model_scale, model_angle, point_vec,
                    patch_desc_size, self.esti_scale_ratio_list,
                    scale_angle_result_list=scale_angle_now)
                scale_list_res.append(np.r_[scale_base, scale_angle_now[0]])
                angle_list_res.append(np.r_[angle_base, scale_angle_now[1]])
                patch_new = patch_new / 128 - 1
                with torch.no_grad():
                    desc_new_now = p_desc.POP_patch_desc(easy_net, patch_new)
                desc_list_res[image_id_] = np.r_[desc_list_res[image_id_], desc_new_now]
                point_list_res[image_id_] = np.r_[point_list_res[image_id_], point_vec_new]

            return point_list_res, desc_list_res, scale_list_res, angle_list_res

        # ################## end the definition of sub-functions ###################
        result_list = []
        with open(self.pair_txt_path + '_seq', 'w') as f:
            pass
        with open(self.pair_txt_path, 'w') as f:
            pass

        for idx, batch in enumerate(dataloader, 0):
            input, H, image_shape, image_name = batch['image'], batch['H'], \
                                                batch['image_shape'], batch['image_name']
            input_gray = batch['image_gray']
            image_ori = batch['image_ori']
            image_name_list = batch['image_name_list']
            image_name_list = [name_now[0] for name_now in image_name_list]
            input = input.view(tuple(np.r_[input.shape[0] * input.shape[1],
                                           input.shape[2:]]))
            H = H.view(tuple(np.r_[H.shape[0] * H.shape[1], H.shape[2:]]))
            image_shape_ori = image_shape.view(tuple(np.r_[image_shape.shape[0] * image_shape.shape[1],
                                                           image_shape.shape[2:]]))
            input_gray = input_gray.view(tuple(np.r_[input_gray.shape[0] * input_gray.shape[1],
                                                     input_gray.shape[2:]]))
            image_ori = image_ori.view(tuple(np.r_[image_ori.shape[0] * image_ori.shape[1],
                                                   image_ori.shape[2:]]))
            H = H.numpy()
            image_shape_ori = image_shape_ori.numpy()
            image_ori = image_ori.numpy()

            scale_list = []
            angle_list = []
            if self.methed_name in ['AffNet_HardNet']:
                point_list, desc_list, scale_list, angle_list = get_AffNet_HardNet_point(input, image_ori)
            elif self.methed_name == 'POP':
                point_list, desc_list, scale_list, angle_list = get_POP_point(input)
            elif self.methed_name == 'AffNet_esti_HardNet':
                point_list, desc_list, scale_list, angle_list = get_AffNet_esti_HardNet_point(input, image_ori)
            elif self.methed_name == 'POP_esti':
                point_list, desc_list, scale_list, angle_list = get_POP_esti_point(input)
            else:
                point_list = None
                desc_list = None
                assert 'unknown method'

            if point_list is None:
                point_list = [np.zeros((1, 2)) for _ in range(image_ori.shape[0])]
                desc_list = [np.ones((1, 64)) / 8 for _ in range(image_ori.shape[0])]

            need_single = False
            if self.methed_name in self.need_single_point_alg:
                need_single = True
            eval_obj = EvaluationSep(point_list, desc_list, H,
                                     self.soft_dist_vec, self.out_dist,
                                     self.image_row, self.image_col, image_shape_ori,
                                     need_single, self.pair_txt_path,
                                     scale_list, angle_list, self.match_image_path)
            result_dict = eval_obj.get_homograghy_esti(
                    write_match_mark=True, image_name=image_name[0],
                    image_ori=image_ori)
            repeat_str = ','.join(('%.4f' % item) for item in result_dict['repeat'])
            mscore_str = ','.join(('%.4f' % item) for item in result_dict['m_score'])
            homo_str = ','.join(('%.3f' % item) for item in result_dict['homo_corr'])
            homo_std_str = ','.join(('%.3f' % item) for item in result_dict['homo_corr_std'])
            result_str = ('%s,HA:,%s,HA_std:,%s,MS:,%s,Rep:,%s,point_num:%.3f' %
                          (image_name[0], homo_str, homo_std_str,
                           mscore_str, repeat_str, result_dict['point_num']))
            print(result_str)
            result_str_sa = ('s_error:%.3f, s_error_std:%.3f, a_error:%.3f, a_error_std:%.3f' %
                             (result_dict['scale_error'], result_dict['scale_error_std'],
                              result_dict['angle_error'], result_dict['angle_error_std']))
            print(result_str_sa)

            with open(self.pair_txt_path + '_seq', 'a') as f:
                f.write('%s,%s\n' % (result_str, result_str_sa))

            result_list.append(result_dict)

        result_mean_dict = {}
        for k_now in result_list[0].keys():
            item_mat = np.array([item[k_now] for item in result_list])
            result_mean_dict[k_now] = np.mean(item_mat, axis=0)

        return result_mean_dict


    def get_point_from_resp(self, resp_tensor):
        resp_max = F.max_pool2d(resp_tensor, kernel_size=2 * self.nms_rad + 1,
                                padding=self.nms_rad, stride=1)
        max_pos_tensor = (resp_max == resp_tensor).transpose(1, 0)
        max_pos_tensor = max_pos_tensor.squeeze(0)

        input_each_num, image_row, image_col = \
            resp_tensor.shape[0], resp_tensor.shape[2], resp_tensor.shape[3]
        xy_range = np.array([self.out_dist, image_col - self.out_dist,
                             self.out_dist, image_row - self.out_dist])

        point_list = []
        for image_id in range(input_each_num):
            resp_now = resp_tensor[image_id].cpu().numpy().squeeze()

            max_pos_now = max_pos_tensor[image_id].cpu().numpy().squeeze().astype('bool')
            max_mask = (max_pos_now & (resp_now > self.resp_thre))
            point_y, point_x = np.where(max_mask)
            point_here = remove_out_point(np.c_[point_x, point_y], xy_range)
            point_loc = point_here[:, 1] * image_col + point_here[:, 0]
            point_resp = resp_now.take(point_loc)
            point_now = np.c_[point_here[:, 0], point_here[:, 1], point_resp]
            point_now, point_ind = self.nms_fast(point_now, self.nms_rad,
                                                 (image_row, image_col))
            point_now, _ = self.select_k_best(point_now, self.max_point_num)
            point_list.append(point_now)

        return point_list

    def calcu_desc_from_map(self, point_list: list, coarse_desc: torch.Tensor):
        image_num = len(point_list)
        desc_list = []
        for image_id in range(image_num):
            coarse_desc_now = coarse_desc[image_id]
            coarse_desc_now = coarse_desc_now.view(tuple(np.r_[1, coarse_desc_now.shape[:]]))
            point_now = point_list[image_id]
            point_num = point_now.shape[0]
            if point_num < 1:
                desc_list.append(np.zeros(0))
                continue

            samp_pts = torch.from_numpy(point_now[:, :2].copy())
            samp_pts[:, 0] = (samp_pts[:, 0] / (self.image_col / 2.)) - 1.
            samp_pts[:, 1] = (samp_pts[:, 1] / (self.image_row / 2.)) - 1.
            samp_pts = samp_pts.contiguous().view(1, 1, -1, 2)
            samp_pts = samp_pts.float()
            samp_pts = samp_pts.to(coarse_desc.device)
            desc = torch.nn.functional.grid_sample(coarse_desc_now, samp_pts)
            desc = desc.data.cpu().numpy().reshape(-1, point_num)
            desc /= np.linalg.norm(desc, axis=0)[np.newaxis, :]
            desc = desc.transpose((1, 0))
            desc_list.append(desc)

        return desc_list

    def get_easy_desc_from_map(self, point_list: list, desc: torch.Tensor):
        image_num = len(point_list)
        desc_list = []
        for image_id in range(image_num):
            point_now = point_list[image_id]
            point_num = point_now.shape[0]
            if point_num < 1:
                desc_list.append(np.zeros(0))
                continue

            point_desc = desc[image_id, :, point_now[:, 1],
                         point_now[:, 0]].transpose(1, 0)

            desc_list.append(point_desc.cpu().numpy())

        return desc_list

    def select_k_best(self, points: np.ndarray, k: int):
        """ Select the k most probable points (and strip their proba).
        points has shape (num_points, 3) where the last coordinate is the proba. """
        if points.size < 1:
            return points, np.zeros(0)
        sort_ind = points[:, 2].argsort()
        sorted_prob = points[sort_ind, :]
        start = min(k, points.shape[0])
        return sorted_prob[-start:, :], sort_ind[-start:]

    def nms_fast(self, points: np.ndarray, dist_thresh: int, image_shape: tuple = None):
        if image_shape is None:
            image_shape = (self.image_row, self.image_col)
        image_row, image_col = image_shape[:]
        grid = np.zeros((image_row, image_col)).astype(int)  # Track NMS data.
        inds = np.zeros((image_row, image_col)).astype(int)  # Store indices of points.
        # Sort by confidence and round to nearest int.
        inds1 = np.argsort(-points[:, 2])
        points = points[inds1, :]
        rpoints = points[:, :2].round().astype(int)  # Rounded corners.
        # Check for edge case of 0 or 1 corners.
        if rpoints.shape[0] == 0:
            return np.zeros((0, 3)).astype(int), np.zeros(0).astype(int)
        if rpoints.shape[0] == 1:
            out = np.c_[rpoints, points[:, 2]].reshape(1, 3)
            return out, np.zeros((1)).astype(int)
        # Initialize the grid.
        for i, rc in enumerate(rpoints):
            grid[rpoints[i, 1], rpoints[i, 0]] = 1
            inds[rpoints[i, 1], rpoints[i, 0]] = i
        # Pad the border of the grid, so that we can NMS points near the border.
        pad = dist_thresh
        grid = np.pad(grid, ((pad, pad), (pad, pad)), mode='constant')
        # Iterate through points, highest to lowest conf, suppress neighborhood.
        count = 0
        for i, rc in enumerate(rpoints):
            # Account for top and left padding.
            pt = (rc[0] + pad, rc[1] + pad)
            if grid[pt[1], pt[0]] == 1:  # If not yet suppressed.
                grid[pt[1] - pad:pt[1] + pad + 1, pt[0] - pad:pt[0] + pad + 1] = 0
                grid[pt[1], pt[0]] = -1
                count += 1
        # Get all surviving -1's and return sorted array of remaining corners.
        keepy, keepx = np.where(grid == -1)
        keepy, keepx = keepy - pad, keepx - pad
        inds_keep = inds[keepy, keepx]
        out = points[inds_keep, :]
        values = out[:, -1]
        inds2 = np.argsort(-values)
        out = out[inds2, :]
        out_inds = inds1[inds_keep[inds2]]
        return out, out_inds
