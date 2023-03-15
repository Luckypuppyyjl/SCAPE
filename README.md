# SCAPE

## Introduction

Official code repository for the paper:  
[**Attend to Now and Past: A Simple Baseline for Category-Agnostic Pose Estimation**]  


### Abstract
Category-Agnostic Pose Estimation (CAPE) aims to
localize keypoints in a query image given few support
images. Prior art connects the keypoints with query features
with a transformer decoder in a DETR-like framework.
However, we find such connection i) sub-optimally uses
the transformer decoder and ii) entails unnecessary
computation in the matching head. In this work, we
present SCAPE, a Simple baseline for CAPE. It uses the
transformer encoder only and adopts a simple MLP head to
regress the keypoint coordinates. Further, we observe that
the early self-attention struggles to establish the relation
among keypoints and thus cannot capture correct keypoint
positions. To speed up the attention process, we introduce
an attention refiner for the keypoint features, which exploits
the similarity both by now and from the past. SCAPE learns
a filter to mask unimportant information in the current
attention map, and the filtered map is then modulated by
the initial similarity map of keypoints to form an attention
guidance to strengthen the synergy among keypoints. On
the MP-100 dataset, compared with the state-of-the-art
method, SCAPE achieves an average improvement of +6.9
and +9.7 under 1-shot and 5-shot settings, respectively,
while with 57% parameters, 55% GFLOPs, and 27%
training memory. Code will be available at https://github.com/Luckypuppyyjl/SCAPE. 

## Usage

### Install

1. Install [mmpose](https://github.com/open-mmlab/mmpose).
2. run `python setup.py develop`.

### Training
You can follow the guideline of [mmpose](https://github.com/open-mmlab/mmpose/blob/master/docs/en/get_started.md).

#### Train with multiple GPUs
```shell
./tools/dist_train.sh ${CONFIG_FILE} ${GPU_NUM} [optional arguments]
```

#### Train with multiple machines

If you can run this code on a cluster managed with [slurm](https://slurm.schedmd.com/), you can use the script `slurm_train.sh`. (This script also supports single machine training.)

```shell
./tools/slurm_train.sh ${PARTITION} ${JOB_NAME} ${CONFIG_FILE} ${WORK_DIR}
```

Here is an example of using 16 GPUs to train POMNet on the dev partition in a slurm cluster.
(Use `GPUS_PER_NODE=8` to specify a single slurm cluster node with 8 GPUs, `CPUS_PER_TASK=2` to use 2 cpus per task.
Assume that `Test` is a valid ${PARTITION} name.)

```shell
GPUS=16 GPUS_PER_NODE=8 CPUS_PER_TASK=2 ./tools/slurm_train.sh Test pomnet \
  configs/mp100/pomnet/pomnet_mp100_split1_256x256_1shot.py \
  work_dirs/pomnet_mp100_split1_256x256_1shot
```


## MP-100 Dataset


### Terms of Use
1. The dataset is only for non-commercial research purposes. 
2. All images of the MP-100 dataset are from existing datasets ([COCO](http://cocodataset.org/), 
[300W](https://ibug.doc.ic.ac.uk/resources/300-W/), 
[AFLW](https://www.tugraz.at/institute/icg/research/team-bischof/lrs/downloads/aflw/), 
[OneHand10K](https://www.yangangwang.com/papers/WANG-MCC-2018-10.html), 
[DeepFashion](http://mmlab.ie.cuhk.edu.hk/projects/DeepFashion/LandmarkDetection.html), 
[AP-10K](https://github.com/AlexTheBad/AP-10K), 
[MacaquePose](http://www.pri.kyoto-u.ac.jp/datasets/macaquepose/index.html), 
[Vinegar Fly](https://github.com/jgraving/DeepPoseKit-Data), 
[Desert Locust](https://github.com/jgraving/DeepPoseKit-Data), 
[CUB-200](http://www.vision.caltech.edu/datasets/cub_200_2011/), 
[CarFusion](http://www.cs.cmu.edu/~ILIM/projects/IM/CarFusion/cvpr2018/index.html), 
[AnimalWeb](https://fdmaproject.wordpress.com/author/fdmaproject/), 
[Keypoint-5](https://github.com/jiajunwu/3dinn)), which are not our property. We are not responsible for the content nor the meaning of these images. 
3. We provide the [annotations](https://drive.google.com/drive/folders/1pzC5uEgi4AW9RO9_T1J-0xSKF12mdj1_?usp=sharing) for training and testing. However, for legal reasons, we do not host the images. Please follow the [guidance](mp100/README.md) to prepare MP-100 dataset.

```

## Acknowledgement

Thanks to:

- [MMPose](https://github.com/open-mmlab/mmpose)

## License

This project is released under the [Apache 2.0 license](LICENSE).
