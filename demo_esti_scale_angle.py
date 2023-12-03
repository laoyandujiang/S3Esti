import argparse
import numpy as np
import os
import os.path as osp
import cv2
import torch
import math
from abso_esti_model.abso_esti_net import EstiNet
from abso_esti_model.abso_esti_utils import get_wrap_patches_multi

def S3Esti_convert_tensor_fun(image_RGB):
    image = image_RGB[np.newaxis].transpose([0, 3, 1, 2])
    tensor=(torch.from_numpy(image).float() - 128) / 129
    return tensor

def load_model(esti_checkpoint_path,device):
    patch_size = 32
    scale_num = 300
    angle_num = 360
    esti_scale_ratio_list = [0.5, 1, 2]
    checkpoint = torch.load(esti_checkpoint_path, map_location=device)
    model_scale = EstiNet(need_bn=True, device=device, out_channels=scale_num,
                                  patch_size=patch_size, scale_ratio=esti_scale_ratio_list)
    model_angle = EstiNet(need_bn=True, device=device, out_channels=angle_num,
                                  patch_size=patch_size, scale_ratio=esti_scale_ratio_list)
    checkpoint['model_scale'].pop('base')
    checkpoint['model_angle'].pop('base')
    model_scale.load_state_dict(checkpoint['model_scale'], strict=True)
    model_scale.eval()
    model_angle.load_state_dict(checkpoint['model_angle'], strict=True)
    model_angle.eval()
    model_scale=model_scale.to(device)
    model_angle=model_angle.to(device)
    
    return model_scale,model_angle,esti_scale_ratio_list

def esti(args):
    image_path = args.image_path
    visual_path = args.save_visualization_path
    device = args.device
    
    # load the estimators
    esti_checkpoint_path = 'abso_esti_model/S3Esti_ep30.pth'
    model_scale,model_angle,esti_scale_ratio_list=load_model(esti_checkpoint_path,device)
    
    # read the testing image
    image = cv2.imread(image_path)
    image_RGB=cv2.cvtColor(image,cv2.COLOR_BGR2RGB)
    
    # SIFT is used as the detector in this demo
    image_gray=cv2.cvtColor(image_RGB,cv2.COLOR_RGB2GRAY)
    sift = cv2.SIFT_create(nfeatures=1000)
    kp_sift = sift.detect(image_gray, None)
    point_vec=np.array([item.pt for item in kp_sift])
    
    # convert the RGB image to a tensor
    image_tensor=S3Esti_convert_tensor_fun(image_RGB)
    image_tensor=image_tensor.to(device)
    
    # In this demo script, "patch_desc_size" is just a placeholder parameter which
    # doesn't influence the estimation results
    patch_desc_size=4
    scale_angle_now = []
    _,point_vec_out=get_wrap_patches_multi(image_tensor, model_scale, model_angle, point_vec,
                    patch_desc_size, esti_scale_ratio_list,
                    scale_angle_result_list=scale_angle_now,scale_num=1,angle_num=3)
    scale_vec_out=scale_angle_now[0]
    angle_vec_out=scale_angle_now[1]
    # organize the result as the "cv2.KeyPoint" format
    point_out_num=point_vec_out.shape[0]
    kp_list=[cv2.KeyPoint() for _ in range(point_out_num)]
    for kp_ind in range(point_out_num):
        kp_list[kp_ind].pt=(point_vec_out[kp_ind,0],point_vec_out[kp_ind,1])
        kp_list[kp_ind].angle=angle_vec_out[kp_ind]*180/math.pi
        # "scale_show_ratio" enlarges the scale values to visualize them with the OpenCV function
        scale_show_ratio=32
        kp_list[kp_ind].size=scale_vec_out[kp_ind]*scale_show_ratio
    
    # visualize the estimation result
    visualization_image=image.copy()
    cv2.drawKeypoints(image, kp_list, visualization_image, 
                      flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
    out_path=osp.join(visual_path,osp.basename(image_path)+'.png')
    cv2.imwrite(out_path,visualization_image)

def main():
    parser = argparse.ArgumentParser(description="The evaluation of POP and compared methods")

    parser.add_argument('image_path', type=str,
                        help='The path of testing image')
    parser.add_argument('--device', type=str, default='cuda:0',
                        help='The device used to perform the model, ' \
                             'which is represented with the pytorch format')
    parser.add_argument('--save_visualization_path', type=str,
                        default='demo_visualization',
                        help='The path to save the image matching results, ' \
                             'which requires some spaces to store the result of every image pair.')

    args = parser.parse_args()

    # create the storage directory if needed
    if not os.path.exists(args.save_visualization_path):
        os.mkdir(args.save_visualization_path)

    esti(args)

if __name__ == '__main__':
    main()
