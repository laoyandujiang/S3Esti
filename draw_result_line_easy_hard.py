import argparse
import matplotlib.pyplot as plt
import warnings
import os
import pickle
import numpy as np
import cv2
import math

from show_result_pair_scale_angle import get_dataset_pair_s_o


def save_line(method_list, x_dict, y_dict, color_dict, mark_name, save_dir,
              dot_line_mark=None):
    fig, ax_plot = plt.subplots(figsize=(3.5, 5))
    for key_now in method_list:
        x_vec = np.array(x_dict[key_now])
        y_vec = np.array(y_dict[key_now])
        if (dot_line_mark is not None) and key_now.find(dot_line_mark) >= 0:
            ax_plot.plot(x_vec, y_vec, label=key_now, color=color_dict[key_now],
                         linestyle='--')
        else:
            ax_plot.plot(x_vec, y_vec, label=key_now, color=color_dict[key_now])

    ax_plot.set_ylim(0, 1)

    plt.xticks(np.arange(1, 6, 1), fontsize=14)
    plt.yticks(fontsize=14)

    handles, labels = ax_plot.get_legend_handles_labels()
    img_name = '%s.png' % (mark_name,)
    img_name = os.path.join(save_dir, img_name)

    plt.legend(handles, labels, loc='upper left', fontsize=11)

    plt.savefig(img_name, dpi=300)

    image = cv2.imread(img_name)
    image = image[130:-60, 10:-90]
    cv2.imwrite(img_name, image)

    plt.close("all")


def load_pair_result(pair_result_path, key_name, thre_num):
    with open(pair_result_path, 'r') as f:
        lines = f.readlines()
    lines = [line.split(',') for line in lines]
    key_pos = lines[0].index(key_name)
    result_dict = {}
    for k in range(len(lines)):
        line_now = lines[k]
        values = [float(line_now[key_pos + item]) for item in range(1, 1 + thre_num)]
        result_dict[line_now[0]] = np.array(values)

    return result_dict


def draw(args):
    key_need = args.metric_mark
    dataset_root = args.HPatches_path
    dataset_name = os.path.basename(dataset_root)
    pair_result_root = args.statistics_save_path
    pair_result_path = os.path.join(pair_result_root, dataset_name)
    img_write_root = args.figure_save_path

    plt.close('all')
    scale_vec, angle_vec, pair_key = get_dataset_pair_s_o(dataset_root)
    pair_num = scale_vec.shape[0]
    # handle the scale smaller than 1
    scale_vec[scale_vec < 1] = (1 / scale_vec[scale_vec < 1])
    # concern the abs of the angle
    angle_vec = np.abs(angle_vec)

    split_steps = 10
    group_num = math.ceil(pair_num / split_steps)
    split_pos = [9]
    split_num = len(split_pos) + 1
    assert split_num == 2

    method_mark = 'S3Esti_AEU'
    method_list = ['POP', 'POP_esti',
                   'AffNet_HardNet', 'AffNet_esti_HardNet']
    method_show_dict = {'POP': 'POP',
                        'POP_esti': 'POP+S3Esti',
                        'AffNet_HardNet': 'HAN_HN',
                        'AffNet_esti_HardNet': 'HAN_HN+S3Esti'}
    method_color = {'POP': 'blue',
                    'POP+S3Esti': 'blue',
                    'HAN_HN': 'red',
                    'HAN_HN+S3Esti': 'red',}

    thre_num = 5
    img_write_root = os.path.join(img_write_root, key_need)
    if not os.path.exists(img_write_root):
        os.mkdir(img_write_root)

    dist_dict_list = []
    scale_dict_list = []
    angle_dict_list = []
    scale_thre = 2
    angle_thre = 20
    both_dict_list = [{}, {}]
    for method_id in range(len(method_list)):
        method_base = method_list[method_id]
        show_name_now = method_show_dict[method_base]
        pair_path = os.path.join(pair_result_path, method_base)
        scale_split_value = np.zeros((split_num, 2))
        angle_split_value = np.zeros((split_num, 2))
        HA_show_scale = np.zeros((split_num, thre_num))
        HA_show_angle = np.zeros((split_num, thre_num))

        HA_dict = load_pair_result(pair_path, key_need + ':', thre_num)
        HA_vec = np.zeros((pair_num, thre_num))
        for idx, key_now in enumerate(pair_key):
            HA_vec[idx] = HA_dict[key_now]

        scale_index = np.argsort(scale_vec)
        scale_sort = scale_vec[scale_index]
        HA_sort_scale = HA_vec[scale_index]
        angle_index = np.argsort(angle_vec)
        angle_sort = angle_vec[angle_index]
        HA_sort_angle = HA_vec[angle_index]
        start_step_now = 0
        for k in range(split_num):
            if k == 0:
                start_ind_scale = 0
                end_ind_scale = np.where(scale_sort <= scale_thre)[0][-1] + 1
                start_ind_angle = 0
                end_ind_angle = np.where(angle_sort <= angle_thre)[0][-1] + 1
            elif k == 1:
                start_ind_scale = np.where(scale_sort > scale_thre)[0][0]
                end_ind_scale = pair_num
                start_ind_angle = np.where(angle_sort > angle_thre)[0][0]
                end_ind_angle = pair_num
            else:
                start_ind_scale = -1
                end_ind_scale = -1
                start_ind_angle = -1
                end_ind_angle = -1
            # print([start_ind_scale, end_ind_scale], [start_ind_angle, end_ind_angle],
            #       (end_ind_scale - start_ind_scale + 1),
            #       (end_ind_angle - start_ind_angle + 1))

            HA_show_scale[k] = np.mean(HA_sort_scale[start_ind_scale:end_ind_scale], axis=0)
            scale_split_value[k, 0] = scale_sort[start_ind_scale]
            scale_split_value[k, 1] = scale_sort[end_ind_scale - 1]
            HA_show_angle[k] = np.mean(HA_sort_angle[start_ind_angle:end_ind_angle], axis=0)
            angle_split_value[k, 0] = angle_sort[start_ind_angle]
            angle_split_value[k, 1] = angle_sort[end_ind_angle - 1]
            if len(dist_dict_list) < k + 1:
                dist_dict_list.append({})
                scale_dict_list.append({})
                angle_dict_list.append({})
            dist_dict_list[k][show_name_now] = []
            scale_dict_list[k][show_name_now] = []
            angle_dict_list[k][show_name_now] = []
            for error_id in range(0, thre_num):
                dist_dict_list[k][show_name_now].append(error_id + 1)
                scale_dict_list[k][show_name_now].append(HA_show_scale[k, error_id])
                angle_dict_list[k][show_name_now].append(HA_show_angle[k, error_id])
            if k == split_num - 1:
                both2_index = np.concatenate([
                    scale_index[start_ind_scale:end_ind_scale],
                    angle_index[start_ind_angle:end_ind_angle]], axis=0)
                both2_index = np.unique(both2_index)
                both1_index = np.setdiff1d(scale_index, both2_index, assume_unique=True)
                HA_show_both = np.zeros((2, thre_num))
                if both1_index.shape[0] < 1:
                    pass
                else:
                    HA_show_both[0] = np.mean(HA_vec[both1_index], axis=0)
                HA_show_both[1] = np.mean(HA_vec[both2_index], axis=0)
                both_dict_list[0][show_name_now] = []
                both_dict_list[1][show_name_now] = []
                for error_id in range(0, thre_num):
                    both_dict_list[0][show_name_now].append(HA_show_both[0, error_id])
                    both_dict_list[1][show_name_now].append(HA_show_both[1, error_id])
                temp = 1

    method_show_list = []
    for method_name in method_list:
        method_show_list.append(method_show_dict[method_name])
    for k in range(split_num):
        if k == split_num - 1:
            save_line(method_show_list, dist_dict_list[k], both_dict_list[-1],
                      method_color, '%s_both_%s_%d' % (method_mark, key_need, k),
                      img_write_root, dot_line_mark='Esti')
        else:
            save_line(method_show_list, dist_dict_list[k], both_dict_list[k],
                      method_color, '%s_both_%s_%d' % (method_mark, key_need, k),
                      img_write_root, dot_line_mark='Esti')


def main():
    parser = argparse.ArgumentParser(description="Draw the accuracy curves")

    parser.add_argument('--HPatches-path', type=str,
                        default='./test_image',
                        help='The path of hpatches sequences dataset')
    parser.add_argument('--statistics-save-path', type=str,
                        default='statistics_results',
                        help='The path to save the statistics results')
    parser.add_argument('--figure-save-path', type=str,
                        default='figure_results',
                        help='The path to save the image matching results, ' \
                             'which requires some spaces to store the result of every image pair.')
    parser.add_argument('--metric-mark', type=str,
                        default='HA', choices=['MS', 'HA'],
                        help='Indicate which metric needs to be shown. '\
                             'MS: matching score, HA: homography accuracy')


    args = parser.parse_args()

    # create the storage directory if needed
    if not os.path.exists(args.figure_save_path):
        os.mkdir(args.figure_save_path)

    draw(args)

if __name__ == '__main__':
    main()
