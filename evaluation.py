import argparse
import numpy as np
import warnings
import os
from evaluation_main import EvaluationMain

warnings.filterwarnings("ignore")


def eval(args):
    each_pair_txt_root = args.statistics_save_path
    match_visual_path = args.matching_image_save_path
    device = args.device
    method_info_list = []

    method_info_now = {}
    method_info_now['method_name'] = 'POP_esti'
    method_info_now['para_dict'] = {'resp_thre': 0.5,
                                    'write_filename': 'POP_esti'}
    method_info_list.append(method_info_now)

    method_info_now = {}
    method_info_now['method_name'] = 'AffNet_esti_HardNet'
    method_info_now['para_dict'] = {'write_filename': 'AffNet_esti_HardNet'}
    method_info_list.append(method_info_now)

    method_info_now = {}
    method_info_now['method_name'] = 'POP'
    method_info_now['para_dict'] = {'resp_thre': 0.5,
                                    'write_filename': 'POP'}
    method_info_list.append(method_info_now)

    method_info_now = {}
    method_info_now['method_name'] = 'AffNet_HardNet'
    method_info_now['para_dict'] = {'write_filename': 'AffNet_HardNet'}
    method_info_list.append(method_info_now)

    dataset_info_list = []
    dataset_info_now = {}
    dataset_info_now['data_dir'] = args.HPatches_path
    dataset_info_now['name'] = os.path.basename(args.HPatches_path)
    dataset_info_now['suffix'] = args.image_ext
    dataset_info_list.append(dataset_info_now)

    image_info_list = []
    image_info_now = {'image_row': 480, 'image_col': 640, 'max_point_num': 1000}
    image_info_list.append(image_info_now)

    soft_dist_vec = np.arange(5) + 1

    for method_id, method_info in enumerate(method_info_list):
        for dataset_id, dataset_info in enumerate(dataset_info_list):
            data_dir = dataset_info['data_dir']
            for image_info in image_info_list:
                image_row, image_col = image_info['image_row'], image_info['image_col']
                max_point_num = image_info['max_point_num']

                nms_rad = 4
                out_dist = 7
                method_ori_name = method_info['method_name']
                method_fullname = method_info['method_name']
                print(method_fullname, '----', dataset_info['name'])
                if 'write_filename' in method_info['para_dict'].keys():
                    method_fullname = method_info['para_dict']['write_filename']

                evaluator = EvaluationMain(device)
                match_image_path = match_visual_path
                if match_visual_path is not None:
                    match_image_path = os.path.join(match_visual_path, method_fullname,
                                                    dataset_info['name'])
                pair_txt_path = os.path.join(
                    each_pair_txt_root, '%s' % (dataset_info['name']))
                if not os.path.exists(pair_txt_path):
                    os.mkdir(pair_txt_path)
                pair_txt_path = os.path.join(pair_txt_path, method_fullname)

                evaluator.set_dataset(data_dir, dataset_info['suffix'],
                                      pair_txt_path, match_image_path=match_image_path)
                evaluator.set_hyper(nms_rad, soft_dist_vec, out_dist, max_point_num,
                                    image_row, image_col)
                result_dict = evaluator.main(method_info['method_name'],
                                             method_info['para_dict'])

                # the basic information of the current evaluation
                soft_dist_str = ', '.join([('%.1f' % soft_dist_vec[pos]) for pos
                                           in range(soft_dist_vec.shape[0])])
                result_filename = os.path.join(each_pair_txt_root, method_fullname + '.txt')
                result_info_str = (
                    'dataset: %s, image_row: %d, image_col: %d, max_point_num: %d,'
                    'soft_dist: %s' %
                    (dataset_info['name'], image_info['image_row'],
                     image_info['image_col'], image_info['max_point_num'], soft_dist_str))

                # construct the string of all the metrics
                result_str = ''
                result_item = []
                soft_dist_num = soft_dist_vec.shape[0]
                for k_pos, key_now in enumerate(result_dict.keys()):
                    result_item.append(key_now)
                    result_now = result_dict[key_now]
                    if isinstance(result_now, np.ndarray):
                        str_now = ','.join(['%.5f' % item for item in result_now])
                    else:
                        str_now = '%.5f' % result_now

                    result_str += (key_now + ': ' + str_now + '\n')

                with open(result_filename, 'a') as file_to_write:
                    file_to_write.write('\n-------------------\n\n')
                    file_to_write.write(result_info_str)
                    file_to_write.write('\n\n')
                    file_to_write.write(result_str)
                    file_to_write.write('\n')

                temp = 1


def main():
    parser = argparse.ArgumentParser(description="The evaluation of POP and compared methods")

    parser.add_argument('--HPatches-path', type=str,
                        default='./test_image',
                        help='The path of hpatches sequences dataset')
    parser.add_argument('--image-ext', type=str,
                        default='.ppm',
                        help='The extension of the evaluation images')
    parser.add_argument('--device', type=str, default='cuda:0',
                        help='The device used to perform the model, ' \
                             'which is represented with the pytorch format')
    parser.add_argument('--statistics-save-path', type=str,
                        default='statistics_results',
                        help='The path to save the statistics results. '
                             'The default path is ./statistics_results')
    parser.add_argument('--matching-image-save-path', type=str,
                        default=None,
                        help='The path to save the image matching results, ' \
                             'which requires some spaces to store the result of every image pair.')

    args = parser.parse_args()

    # create the storage directory if needed
    if not os.path.exists(args.statistics_save_path):
        os.mkdir(args.statistics_save_path)
    if args.matching_image_save_path is not None:
        if not os.path.exists(args.matching_image_save_path):
            os.mkdir(args.matching_image_save_path)

    eval(args)


if __name__ == '__main__':
    main()
