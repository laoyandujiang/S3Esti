import argparse
import torch
import cv2
import math
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from abso_esti_model.training_dataset import ImageSet
from abso_esti_model.abso_esti_net import EstiNet
import numpy as np
import torch.optim as optim
import random
import time
import os
import torch.nn.functional as F
import copy


def train(args):
    image_set_path = args.training_path
    checkpoint_dir = args.save_checkpoint_path
    status_txt_name = args.training_status_txt_name
    device = torch.device(args.device)
    batch_size = args.batch_size
    checkpoint_name = args.restore_checkpoint_path

    image_channels = 3
    patch_size = 32
    patch_num = 32
    scale_num = 300
    angle_num = 360
    # the multiple-scale patches are used as the training input
    scale_ratio_list = [0.5, 1, 2]

    model_scale = EstiNet(need_bn=True, device=device, out_channels=scale_num,
                          patch_size=patch_size, scale_ratio=scale_ratio_list)
    model_angle = EstiNet(need_bn=True, device=device, out_channels=angle_num,
                          patch_size=patch_size, scale_ratio=scale_ratio_list)

    use_pretrain = (checkpoint_name is not None)
    if use_pretrain:
        checkpoint = torch.load(checkpoint_name, map_location=device)
        model_scale.load_state_dict(checkpoint['model_scale'], strict=True)
        model_scale.train()
        model_angle.load_state_dict(checkpoint['model_angle'], strict=True)
        model_angle.train()
    model_scale.to(device)
    model_angle.to(device)

    # objective
    CE_loss = torch.nn.CrossEntropyLoss()
    optimizer_scale = optim.SGD(model_scale.parameters(), lr=0.001, momentum=0.9)
    optimizer_angle = optim.SGD(model_angle.parameters(), lr=0.001, momentum=0.9)
    if use_pretrain:
        optimizer_scale.load_state_dict(checkpoint['optimizer_scale'])
        optimizer_angle.load_state_dict(checkpoint['optimizer_angle'])

    # set the dataloader
    scale_limit = [1.0, 3.0]
    dataset = ImageSet(image_set_path, patch_size, patch_num,
                       scale_limit, model_angle.angle_limit,
                       is_train=True, flatten_dir=True)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=4)
    iter_num_in_epoch = len(dataloader)

    # saving parameter
    print_inter = 4
    save_inter = 400
    save_checkpoint_inter = 2000
    write_txt_inter = 50
    epoch_max = 99999999
    if not use_pretrain:
        with open(status_txt_name, 'w'):
            pass

    # training hyperparameter
    check_stable_iter_min = max(5000, iter_num_in_epoch)
    check_stable_iter_max = check_stable_iter_min + 500
    check_stable_angle_thre = 1.53
    angle_error_cum = None
    unstable_mark = False

    # start running
    cum_iter_num = 0
    last_checkpoint_num = 0
    epoch = 0
    if use_pretrain:
        epoch = checkpoint['epoch'] + 1
    while epoch < epoch_max:
        for idx, batch in enumerate(dataloader):
            cum_iter_num += 1
            last_checkpoint_num += 1

            optimizer_scale.zero_grad()
            optimizer_angle.zero_grad()
            # the data in the current batch
            patch1, patch2, scale, angle = \
                batch['patch1'], batch['patch2'], batch['scale'], batch['angle']
            batch_size = patch1.shape[0]
            patch1 = batch['patch1'].view(tuple(np.r_[batch_size * patch1.shape[1],
                                                      patch1.shape[2:]]))
            patch2 = batch['patch2'].view(tuple(np.r_[batch_size * patch2.shape[1],
                                                      patch2.shape[2:]]))
            scale12 = batch['scale'].view((batch_size * scale.shape[1], 1))
            angle12 = batch['angle'].view((batch_size * angle.shape[1], 1))

            patch1 = patch1.to(device)
            patch2 = patch2.to(device)
            scale12 = scale12.to(device)
            angle12 = angle12.to(device)

            patch_num_now = patch1.shape[0]
            scale1_resp = model_scale(patch1)
            angle1_resp = model_angle(patch1)
            scale2_resp = model_scale(patch2)
            angle2_resp = model_angle(patch2)
            # obtain the labels of the relative scale and orientation
            scale12_label = model_scale.map_scale_rela_to_id(scale12)
            angle12_label = model_scale.map_angle_to_id(angle12)
            # obtain the absolute labels by minimize the current loss
            with torch.no_grad():
                scale1_label, angle1_label, scale2_label, angle2_label = \
                    model_scale.get_max_label(scale1_resp, angle1_resp, scale2_resp,
                                              angle2_resp, scale12, angle12)
            # compute the current estimation error
            with torch.no_grad():
                # validate the correctness of the relative labels (helpful to debug)
                scale_rela_label_error, angle_rela_label_error, _, _ = model_scale.get_rela_pred_error(
                    scale12_label, angle12_label, scale12, angle12)
                # compute the current estimation error between the predictions and labels
                scale1_ind_pred = torch.argmax(scale1_resp, dim=1)
                angle1_ind_pred = torch.argmax(angle1_resp, dim=1)
                scale2_ind_pred = torch.argmax(scale2_resp, dim=1)
                angle2_ind_pred = torch.argmax(angle2_resp, dim=1)
                scale_error, angle_error, _, _, _, _ = model_scale.get_pred_error(
                    scale1_ind_pred, angle1_ind_pred, scale2_ind_pred,
                    angle2_ind_pred, scale12, angle12)
                # validate the correctness of the absolute labels (helpful to debug)
                scale_label_error, angle_label_error, _, _, _, _ = model_scale.get_pred_error(
                    scale1_label, angle1_label, scale2_label, angle2_label,
                    scale12, angle12)

            # compute the classification loss
            scale1_label, angle1_label, scale2_label, angle2_label = \
                (scale1_label.to(device), angle1_label.to(device),
                 scale2_label.to(device), angle2_label.to(device))
            loss_scale = CE_loss(scale1_resp, scale1_label) + CE_loss(scale2_resp, scale2_label)
            loss_angle = CE_loss(angle1_resp, angle1_label) + CE_loss(angle2_resp, angle2_label)

            loss = loss_scale + loss_angle

            # update the network parameters
            loss.backward()
            optimizer_scale.step()
            optimizer_angle.step()

            assert (max((scale_label_error, angle_label_error, scale_rela_label_error,
                         angle_rela_label_error)) < 0.01)
            str = ('ep:%d, iter:%d/%d, l:%.3f, l_s:%.3f, l_a:%.3f, s_e:%.4f, a_e:%.4f, '
                   's_l_e:%.4f, a_l_e:%.4f' % (
                       epoch, idx, iter_num_in_epoch, loss.item(), loss_scale.item(), loss_angle.item(),
                       scale_error, angle_error,
                       scale_label_error, angle_label_error))

            if cum_iter_num % print_inter == 0:
                print(str)
            if (cum_iter_num % write_txt_inter) == 0:
                with open(status_txt_name, 'a') as f:
                    f.write(str + '\n')
            if cum_iter_num % save_inter == save_inter - 1:
                torch.save({
                    'model_scale': model_scale.state_dict(),
                    'optimizer_scale': optimizer_scale.state_dict(),
                    'model_angle': model_angle.state_dict(),
                    'optimizer_angle': optimizer_angle.state_dict(),
                    'epoch': epoch,
                }, os.path.join(checkpoint_dir, 'checkpoint.pth'))

            # check whether the training converges stably
            if cum_iter_num == check_stable_iter_min:
                angle_error_cum = 0
            if check_stable_iter_min <= cum_iter_num < check_stable_iter_max:
                angle_error_cum += angle_error
            if cum_iter_num == check_stable_iter_max:
                angle_error_mean = angle_error_cum / (check_stable_iter_max - check_stable_iter_min)
                if angle_error_mean > check_stable_angle_thre:
                    # the current training doesn't converge well
                    # initialize all the configurations and re-train the model
                    cum_iter_num = 0
                    last_checkpoint_num = 0
                    epoch = 0
                    model_scale = EstiNet(need_bn=True, device=device, out_channels=scale_num,
                                          patch_size=patch_size, scale_ratio=scale_ratio_list)
                    model_angle = EstiNet(need_bn=True, device=device, out_channels=angle_num,
                                          patch_size=patch_size, scale_ratio=scale_ratio_list)
                    model_scale.to(device)
                    model_angle.to(device)
                    optimizer_scale = optim.SGD(model_scale.parameters(), lr=0.001, momentum=0.9)
                    optimizer_angle = optim.SGD(model_angle.parameters(), lr=0.001, momentum=0.9)
                    dataset = ImageSet(image_set_path, patch_size, patch_num,
                                       scale_limit, model_angle.angle_limit,
                                       is_train=True, flatten_dir=True)
                    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=4)
                    with open(status_txt_name, 'w'):
                        pass
                    unstable_mark = True
                    break

        if unstable_mark:
            unstable_mark = False
            continue
        # save the checkpoint model when an epoch is completed
        # and there are enough iterations after the last saving
        if last_checkpoint_num > save_checkpoint_inter:
            torch.save({
                'model_scale': model_scale.state_dict(),
                'optimizer_scale': optimizer_scale.state_dict(),
                'model_angle': model_angle.state_dict(),
                'optimizer_angle': optimizer_angle.state_dict(),
                'epoch': epoch,
            }, os.path.join(checkpoint_dir, 'checkpoint_end_ep_%d.pth' % epoch))
            last_checkpoint_num = 0
        epoch += 1


def main():
    parser = argparse.ArgumentParser(description="The evaluation of POP and compared methods")

    parser.add_argument('--training-path', type=str,
                        default='./demo_training_images',
                        help='The path of training dataset')
    parser.add_argument('--save-checkpoint-path', type=str,
                        default='./S3Esti_checkpoint',
                        help='The path to save the checkpoint models')
    parser.add_argument('--restore-checkpoint-path', type=str,
                        default=None,
                        help='set the path of checkpoint if needed, ' \
                             'and use the default when train the model from scratch')
    parser.add_argument('--device', type=str, default='cuda:0',
                        help='The device used to train the model, ' \
                             'which is represented with the pytorch format')
    parser.add_argument('--batch-size', type=int, default=16,
                        help='Training batch size')
    parser.add_argument('--training-status-txt-name', type=str,
                        default='train_status.txt',
                        help='The text filename to record the training loss '
                             'and other status.')

    args = parser.parse_args()

    # create the storage directory if needed
    if not os.path.exists(args.save_checkpoint_path):
        os.mkdir(args.save_checkpoint_path)

    train(args)


if __name__ == '__main__':
    main()
