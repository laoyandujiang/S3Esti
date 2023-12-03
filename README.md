# S3Esti (Soft Self-Supervised Estimator)
This repository contains the implementation of the following paper: Learning Soft Estimator of Keypoint Scale and Orientation with Probabilistic Covariant Loss.

## Getting started

This code is developed with Python 3.8 and PyTorch 1.4, but the packages with the later versions should be also suitable. Typically, conda and pip are the recommended ways to configure the environment:
```bash
conda create -n S3Esti python=3.8
conda activate S3Esti
conda install numpy matplotlib scipy
conda install pytorch==1.4.0 cudatoolkit=10.1 -c pytorch
pip install imgaug
```

Please clone the repository with the git command:

```bash
git clone https://github.com/laoyandujiang/S3Esti.git
``````

The trained weights of S3Esti have been also included in this repository to simplify the configuration.

## Demo: Visualize the scales and orientations of S3Esti
A simple demo script `demo_esti_scale_angle.py` is provided to estimate and visualize the keypoint scales and orientations for a given image. Please run the demo with the following command:

```bash
python demo_esti_scale_angle.py xxx/xxx.png
``````

Here, `xxx/xxx.png` is the path of the testing image. The demo script will localize keypoints with a SIFT detector, and then estimate and visualize their scales and orientations.

## Evaluation of S3Esti
We provide the evaluation code for [HPatches](https://github.com/hpatches/hpatches-dataset) dataset, which references the evaluation processes of [HesAffNet](https://github.com/ducha-aiki/affnet) and [POP](https://github.com/elvintanhust/POP-Interest-Point). Before perform the evaluation on the entire HPatches, you can first verify the environment by running the script directly:
```bash
python evaluation.py
```
If the environment is configured correctly, the following information will be printed in the terminal:
```text
AffNet_esti_HardNet ---- test_image
v_boat,HA: ...
s_error: ...
AffNet_HardNet ---- test_image
...
```
Here HesAffNet_HardNet+S3Esti, HesAffNet_HardNet, POP+S3Esti and POP are evaluated on the `v_boat` sequence in HPatches. Note that the `v_boat` sequence has been placed in the `test_image` folder so that the `evaluation.py` script can be run directly. Furthermore, the pre-trained models of [HesAffNet](https://github.com/ducha-aiki/affnet) and [POP](https://github.com/elvintanhust/POP-Interest-Point) are also included in this repository to simplify the configuration.

In the above process, the `statistics_results` folder is created automatically and the main statistics results are written in it. After the evaluations of all methods, four text files, namely `AffNet_esti_HardNet.txt`, `POP_esti.txt`,  `AffNet_HardNet.txt`,  `POP.txt`, should appear in this folder. 

The environment is verified to be correct if the above statistics results can be outputted. Then you can place other data in the `hpatches-sequences-release` folder to further evaluate the methods. Note the format of the data should be consistent to the HPatches sequences. One convenient way is to download the [HPatches](https://github.com/hpatches/hpatches-dataset) sequences first, and then unzip it into the `hpatches-sequences-release` folder.

For more details about all parameters of `evaluation.py`, run `python evaluation.py --help`.

## Drawing the accuracy curves
Before drawing the accuracy curves, the evaluation in the last step should have been finished. Then the accuracy curves can be obtained by running the script:
```bash
python draw_result_line_easy_hard.py
```
The accuracy curves will appear in the `figure_results/` folder. The default curves are drawn for the homography accuracy metric. For more details about all parameters of `draw_result_line_easy_hard.py`, run `python draw_result_line_easy_hard.py --help`.

## Training the model

You can first verify the environment by running the script directly:

```bash
python train.py
```

This command performs the training process, and the training images are in the `demo_input_images/` folder. If the environment is configured correctly, the following information will be printed in the terminal:

```text
ep:3, iter:0/1, l:x.xxx, l_s:x.xxx, l_a:x.xxx, s_e:x.xxxx, a_e:x.xxxx, ...
ep:7, iter:0/1, l:x.xxx, l_s:x.xxx, l_a:x.xxx, s_e:x.xxxx, a_e:x.xxxx, ...
...
```

And the training process will write the checkpoints into the `S3Esti_checkpoint/` folder. The name of the checkpoint is formatted as `checkpoint_end_ep_x`.

The environment is verified to be correct if the above process can be finished without error. To train your model, you can place other data in the `demo_input_images` folder, or specify the training folder with the `--train-image-path` parameter: 

```bash
python train.py --training-path /the/path/of/training/dataset
```

To reproduce the performance in the paper, you can train the model with [COCO 2014 training set](http://images.cocodataset.org/zips/train2014.zip) (containing 82783 images). In our experiments, about 30 epochs are generally required to achieve the performance similar to that in the paper.

You can also set the `--restore-checkpoint-path` parameter to make the model be initialized with the given checkpoint:

```bash
python train.py --restore-checkpoint-path /the/path/of/checkpoint
```

For more details about all parameters of `train.py`, run `python train.py --help`.