import matplotlib
import seaborn as sns
import os
import errno
import numpy as np
from PIL import Image
import sys
from copy import deepcopy
import argparse
import math
import torch.utils.data as data
import torch
import torch.nn.init
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torchvision.datasets as dset
import gc
import torchvision.transforms as transforms
from torch.autograd import Variable
import torch.backends.cudnn as cudnn
from tqdm import tqdm
import random
import cv2
import copy
from Utils import L2Norm, cv2_scale
from Utils import str2bool
from dataset import HPatchesDM, TripletPhotoTour, TotalDatasetsLoader
from augmentation import get_random_norm_affine_LAFs, get_random_rotation_LAFs, get_random_shifts_LAFs
from LAF import denormalizeLAFs, LAFs2ell, abc2A, extract_patches, normalizeLAFs
from pytorch_sift import SIFTNet
from HardNet import HardNet, L2Norm, HardTFeatNet
from Losses import loss_HardNegC, loss_HardNet
from SparseImgRepresenter import ScaleSpaceAffinePatchExtractor
from LAF import denormalizeLAFs, LAFs2ell, abc2A, visualize_LAFs
from Losses import distance_matrix_vector
from ReprojectionStuff import get_GT_correspondence_indexes
from architectures import AffNetFast, AffNetFastScale, AffNetFast4, AffNetFast4RotNosc, AffNetFast52RotUp, \
    AffNetFast52Rot, AffNetFast5Rot, AffNetFast4Rot, AffNetFast4Rot
from architectures import AffNetFast2Par, AffNetFastBias

matplotlib.use('Agg')
# from Utils import np_reshape64 as np_reshape
np_reshape = lambda x: np.reshape(x, (64, 64, 1))
cv2_scale40 = lambda x: cv2.resize(x, dsize=(40, 40),
                                   interpolation=cv2.INTER_LINEAR)

def load_grayscale_var(fname):
    img = Image.open(fname).convert('RGB')
    img = np.mean(np.array(img), axis=2)
    var_image = torch.autograd.Variable(torch.from_numpy(img.astype(np.float32)), volatile=True)
    var_image_reshape = var_image.view(1, 1, var_image.size(0), var_image.size(1))
    var_image_reshape = var_image_reshape.cuda()
    return var_image_reshape


def get_geometry_and_descriptors(img, det, desc, do_ori=True):
    with torch.no_grad():
        LAFs, resp = det(img, do_ori=do_ori)
        patches = det.extract_patches_from_pyr(LAFs, PS=32)
        descriptors = desc(patches)
    return LAFs, descriptors


def test(model, epoch):
    point_num = 3000

    LOG_DIR = 'log'
    if not os.path.isdir(LOG_DIR):
        os.makedirs(LOG_DIR)

    torch.cuda.empty_cache()
    # switch to evaluate mode
    model.eval()
    detector = ScaleSpaceAffinePatchExtractor(mrSize=5.192, num_features=point_num,
                                              border=5, num_Baum_iters=1,
                                              AffNet=model)
    descriptor = HardNet()
    model_weights = 'HardNet++.pth'
    hncheckpoint = torch.load(model_weights)
    descriptor.load_state_dict(hncheckpoint['state_dict'])
    descriptor.eval()
    detector = detector.cuda()
    descriptor = descriptor.cuda()
    input_img_fname1 = 'test-graf/img1.png'  # sys.argv[1]
    input_img_fname2 = 'test-graf/img6.png'  # sys.argv[1]
    H_fname = 'test-graf/H1to6p'  # sys.argv[1]
    output_img_fname = 'graf_match.png'  # sys.argv[3]
    img1 = load_grayscale_var(input_img_fname1)
    img2 = load_grayscale_var(input_img_fname2)
    # RGB image to show result
    img1_RGB = cv2.cvtColor(cv2.imread(input_img_fname1),cv2.COLOR_BGR2RGB)
    img2_RGB = cv2.cvtColor(cv2.imread(input_img_fname2),cv2.COLOR_BGR2RGB)

    H = np.loadtxt(H_fname)
    H1to2 = Variable(torch.from_numpy(H).float())
    SNN_threshold = 0.8
    with torch.no_grad():
        LAFs1, descriptors1 = get_geometry_and_descriptors(img1, detector, descriptor)
        torch.cuda.empty_cache()
        LAFs2, descriptors2 = get_geometry_and_descriptors(img2, detector, descriptor)
        visualize_LAFs(img1_RGB, LAFs1.detach().cpu().numpy().squeeze(), 'b', show=False,
                       save_to=LOG_DIR + "/detections1_" + str(epoch) + '.png')
        visualize_LAFs(img2_RGB, LAFs2.detach().cpu().numpy().squeeze(), 'g', show=False,
                       save_to=LOG_DIR + "/detection2_" + str(epoch) + '.png')

        dist_matrix = distance_matrix_vector(descriptors1, descriptors2)
        min_dist, idxs_in_2 = torch.min(dist_matrix, 1)
        # mask out nearest neighbour to find second nearest
        dist_matrix[:, idxs_in_2] = 100000
        min_2nd_dist, idxs_2nd_in_2 = torch.min(dist_matrix, 1)
        mask = (min_dist / (min_2nd_dist + 1e-8)) <= SNN_threshold
        tent_matches_in_1 = indxs_in1 = \
            torch.autograd.Variable(torch.arange(0, idxs_in_2.size(0)), requires_grad=False).cuda()[mask]
        tent_matches_in_2 = idxs_in_2[mask]
        tent_matches_in_1 = tent_matches_in_1.long()
        tent_matches_in_2 = tent_matches_in_2.long()
        LAF1s_tent = LAFs1[tent_matches_in_1, :, :]
        LAF2s_tent = LAFs2[tent_matches_in_2, :, :]
        min_dist, plain_indxs_in1, idxs_in_2 = get_GT_correspondence_indexes(LAF1s_tent, LAF2s_tent, H1to2.cuda(),
                                                                             dist_threshold=6)
        plain_indxs_in1 = plain_indxs_in1.long()
        inl_ratio = float(plain_indxs_in1.size(0)) / float(tent_matches_in_1.size(0))
        print('Test epoch', str(epoch))
        print('Test on graf1-6,', tent_matches_in_1.size(0), 'tentatives', plain_indxs_in1.size(0), 'true matches',
              str(inl_ratio)[:5], ' inl.ratio')
        visualize_LAFs(img1_RGB,
                       LAF1s_tent[plain_indxs_in1.long(), :, :].detach().cpu().numpy().squeeze(), 'g', show=False,
                       save_to=LOG_DIR + "/inliers1_" + str(epoch) + '.png')
        visualize_LAFs(img2_RGB,
                       LAF2s_tent[plain_indxs_in1.long(), :, :].detach().cpu().numpy().squeeze(), 'g', show=False,
                       save_to=LOG_DIR + "/inliers2_" + str(epoch) + '.png')
        ################
        print('Now native ori')
        # if 'Rot' not in args.arch:
        #    return
        del LAFs1, descriptors1, LAFs2, descriptors2, dist_matrix, tent_matches_in_2, plain_indxs_in1, tent_matches_in_1, idxs_in_2, mask, min_2nd_dist, idxs_2nd_in_2, min_dist, LAF1s_tent, LAF2s_tent
        torch.cuda.empty_cache()
        gc.collect()
        LAFs1, descriptors1 = get_geometry_and_descriptors(img1, detector, descriptor, False)
        LAFs2, descriptors2 = get_geometry_and_descriptors(img2, detector, descriptor, False)
        visualize_LAFs(img1_RGB, LAFs1.detach().cpu().numpy().squeeze(), 'b', show=False,
                       save_to=LOG_DIR + "/ori_detections1_" + str(epoch) + '.png')
        visualize_LAFs(img2_RGB, LAFs2.detach().cpu().numpy().squeeze(), 'g', show=False,
                       save_to=LOG_DIR + "/ori_detection2_" + str(epoch) + '.png')
        dist_matrix = distance_matrix_vector(descriptors1, descriptors2)
        min_dist, idxs_in_2 = torch.min(dist_matrix, 1)
        # mask out nearest neighbour to find second nearest
        dist_matrix[:, idxs_in_2] = 100000
        min_2nd_dist, idxs_2nd_in_2 = torch.min(dist_matrix, 1)
        mask = (min_dist / (min_2nd_dist + 1e-8)) <= SNN_threshold
        tent_matches_in_1 = indxs_in1 = \
            torch.autograd.Variable(torch.arange(0, idxs_in_2.size(0)), requires_grad=False).cuda()[mask]
        tent_matches_in_2 = idxs_in_2[mask]
        tent_matches_in_1 = tent_matches_in_1.long()
        tent_matches_in_2 = tent_matches_in_2.long()
        LAF1s_tent = LAFs1[tent_matches_in_1, :, :]
        LAF2s_tent = LAFs2[tent_matches_in_2, :, :]
        min_dist, plain_indxs_in1, idxs_in_2 = get_GT_correspondence_indexes(LAF1s_tent, LAF2s_tent, H1to2.cuda(),
                                                                             dist_threshold=6)
        plain_indxs_in1 = plain_indxs_in1.long()
        inl_ratio = float(plain_indxs_in1.size(0)) / float(tent_matches_in_1.size(0))
        print('Test epoch', str(epoch))
        print('Test on ori graf1-6,', tent_matches_in_1.size(0),
              'tentatives', plain_indxs_in1.size(0), 'true matches',
              str(inl_ratio)[:5], ' inl.ratio')
        visualize_LAFs(img1_RGB,
                       LAF1s_tent[plain_indxs_in1.long(), :, :].detach().cpu().numpy().squeeze(), 'g', show=False,
                       save_to=LOG_DIR + "/ori_inliers1_" + str(epoch) + '.png')
        visualize_LAFs(img2_RGB,
                       LAF2s_tent[idxs_in_2.long(), :, :].detach().cpu().numpy().squeeze(), 'g', show=False,
                       save_to=LOG_DIR + "/ori_inliers2_" + str(epoch) + '.png')
    return


if __name__ == '__main__':
    detector_path = 'AffNet_det.pth'
    descriptor_path = 'HardNet++.pth'

    PS = 32
    tilt_schedule = {'0': 3.0, '1': 4.0, '3': 4.5,
                     '5': 4.8, '6': 5.2, '8': 5.8}
    detector = AffNetFast(PS=PS)
    checkpoint = torch.load(detector_path)
    start_epoch = checkpoint['epoch']
    detector.load_state_dict(checkpoint['state_dict'], strict=True)
    test(detector, start_epoch)
