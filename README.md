# DETReg: Unsupervised Pretraining with Region Priors for Object Detection
### [Amir Bar](https://amirbar.net), [Xin Wang](https://xinw.ai/), [Vadim Kantorov](http://vadimkantorov.com/), [Colorado J Reed](https://people.eecs.berkeley.edu/~cjrd/), [Roei Herzig](https://roeiherz.github.io/), [Gal Chechik](https://chechiklab.biu.ac.il/), [Anna Rohrbach](https://anna-rohrbach.net/), [Trevor Darrell](https://people.eecs.berkeley.edu/~trevor/), [Amir Globerson](http://www.cs.tau.ac.il/~gamir/)
![DETReg](./figs/illustration.png)
  

This repository is the implementation of DETReg, see [Project Page](https://amirbar.net/detreg).

## Introduction

Recent self-supervised pretraining methods for object detection largely focus on pretraining the backbone of the object detector, neglecting key parts of detection architecture. Instead, we introduce DETReg, a new self-supervised method that pretrains the entire object detection network, including the object localization and embedding components. During pretraining, DETReg predicts object localizations to match the localizations from an unsupervised region proposal generator and simultaneously aligns the corresponding feature embeddings with embeddings from a self-supervised image encoder. We implement DETReg using the DETR family of detectors and show that it improves over competitive baselines when finetuned on COCO, PASCAL VOC, and Airbus Ship benchmarks. In low-data regimes, including semi-supervised and few-shot learning settings, DETReg establishes many state-of-the-art results, e.g., on COCO we see a +6.0 AP improvement for 10-shot detection and +3.5 AP improvement when training with only 1% of the labels.

## Demo

Interact with the DETReg pretrained model in a [Google Colab](https://colab.research.google.com/drive/1ByFXJClyzNVelS7YdT53_bMbwYeMoeNb?usp=sharing)! 

## Installation

### Requirements

* Linux, CUDA>=9.2, GCC>=5.4
  
* Python>=3.7

    We recommend you to use Anaconda to create a conda environment:
    ```bash
    conda create -n detreg python=3.7 pip
    ```
    Then, activate the environment:
    ```bash
    conda activate detreg
    ```
    Installation: (change cudatoolkit to your cuda version. For detailed pytorch installation instructions click [here](https://pytorch.org/))
    ```bash
    conda install pytorch==1.8.0 torchvision==0.9.0 torchaudio==0.8.0 cudatoolkit=10.2 -c pytorch
    ```
  
* Other requirements
    ```bash
    pip install -r requirements.txt
    ```

### Compiling CUDA operators
```bash
cd ./models/ops
sh ./make.sh
# unit test (should see all checking is True)
python test.py
```

## Usage

### Dataset preparation

#### ImageNet/ImageNet100
Download [ImageNet](https://image-net.org/challenges/LSVRC/2012/) and organize it in the following structure:

```
code_root/
└── data/
    └── ilsvrc/
          ├── train/
          └── val/
```
Note that in this work we also used the ImageNet100 dataset, which is x10 smaller than ImageNet. To create ImageNet100 run the following command:
```bash
mkdir -p data/ilsvrc100/train
mkdir -p data/ilsvrc100/val
code_root=/path/to/code_root
while read line; do ln -s "${code_root}/data/ilsvrc/train/$line" ${code_root}/data/ilsvrc100/train/$line"; done < "${code_root}/datasets/category.txt"
while read line; do ln -s "${code_root}/data/ilsvrc/val/$line" "${code_root}/data/ilsvrc100/val/$line"; done < "${code_root>/datasets/category.txt"
```

This should results with the following structure:
```
code_root/
└── data/
    ├── ilsvrc/
          ├── train/
          └── val/
    └── ilsvrc100/
          ├── train/
          └── val/
```

#### MSCoco
Please download [COCO 2017 dataset](https://cocodataset.org/) and organize it in the following structure:

```
code_root/
└── data/
    └── MSCoco/
        ├── train2017/
        ├── val2017/
        └── annotations/
        	├── instances_train2017.json
        	└── instances_val2017.json
```
#### Pascal VOC
Download [Pascal VOC](http://host.robots.ox.ac.uk/pascal/VOC/) dataset (2012trainval, 2007trainval, and 2007test):
```bash
mkdir -p data/pascal
cd data/pascal
wget http://host.robots.ox.ac.uk:8080/pascal/VOC/voc2012/VOCtrainval_11-May-2012.tar
wget http://host.robots.ox.ac.uk/pascal/VOC/voc2007/VOCtrainval_06-Nov-2007.tar
wget http://host.robots.ox.ac.uk/pascal/VOC/voc2007/VOCtest_06-Nov-2007.tar
tar -xvf VOCtrainval_11-May-2012.tar
tar -xvf VOCtrainval_06-Nov-2007.tar
tar -xvf VOCtest_06-Nov-2007.tar
```
The files should be organized in the following structure:
```
code_root/
└── data/
    └── pascal/
        └── VOCdevkit/
        	├── VOC2007
        	└── VOC2012
```

### Pretraining on ImageNet

The command for pretraining DETReg, based on Deformable-DETR, on 8 GPUs on ImageNet is as follows:
```bash
GPUS_PER_NODE=8 ./tools/run_dist_launch.sh 8 ./configs/DETReg_top30_in.sh --batch_size 24 --num_workers 8
```
Using underlying DETR architecture:
```bash
GPUS_PER_NODE=8 ./tools/run_dist_launch.sh 8 ./configs/DETReg_top30_in_detr.sh --batch_size 24 --num_workers 8
```

The command for pretraining DETReg on 8 GPUs on ImageNet100 is as following:
```bash
GPUS_PER_NODE=8 ./tools/run_dist_launch.sh 8 ./configs/DETReg_top30_in100.sh --batch_size 24 --num_workers 8
```
Training takes around 1.5 days with 8 NVIDIA V100 GPUs, you can download a pretrained model (see below) if you want to skip this step.

After pretraining, a checkpoint is saved in ```exps/DETReg_top30_in/checkpoint.pth```. To fine tune it over different coco settings use the following commands:

### Pretraining on MSCoco
The command for pretraining DETReg on 8 GPUs on MSCoco is as following:
```bash
GPUS_PER_NODE=8 ./tools/run_dist_launch.sh 8 ./configs/DETReg_top30_coco.sh --batch_size 24 --num_workers 8
```


### Finetuning on MSCoco from ImageNet pretraining

Fine tuning on full COCO (should take 2 days with 8 NVIDIA V100 GPUs):
```bash
GPUS_PER_NODE=8 ./tools/run_dist_launch.sh 8 ./configs/DETReg_fine_tune_full_coco.sh
```

This assumes a checkpoint exists in `exps/DETReg_top30_in/checkpoint.pth`.

### Finetuning on MSCoco low-data regime, from full MSCoco pretraining (Semi-Supervised Learning setting)

Fine tuning on 1%
```bash
GPUS_PER_NODE=8 ./tools/run_dist_launch.sh 8 ./configs/DETReg_fine_tune_1pct_coco.sh --batch_size 3
```
Fine tuning on 2%
```bash
GPUS_PER_NODE=8 ./tools/run_dist_launch.sh 8 ./configs/DETReg_fine_tune_2pct_coco.sh --batch_size 3
```
Fine tuning on 5%
```bash
GPUS_PER_NODE=8 ./tools/run_dist_launch.sh 8 ./configs/DETReg_fine_tune_5pct_coco.sh --batch_size 3
```
Fine tuning on 10%
```bash
GPUS_PER_NODE=8 ./tools/run_dist_launch.sh 8 ./configs/DETReg_fine_tune_10pct_coco.sh --batch_size 3
```

### Finetuning on Pascal VOC
Fine tune on full Pascal:
```bash
GPUS_PER_NODE=8 ./tools/run_dist_launch.sh 8 ./configs/DETReg_fine_tune_full_pascal.sh --batch_size 4 --epochs 100 --lr_drop 70
```
Fine tune on 10% of Pascal:
```bash
GPUS_PER_NODE=2 ./tools/run_dist_launch.sh 2 ./configs/DETReg_fine_tune_10pct_pascal.sh --batch_size 4 --epochs 200 --lr_drop 150
```



### Evaluation

To evaluate a finetuned model, use the following command from the project basedir:

```bash
./configs/<config file>.sh --resume exps/<config file>/checkpoint.pth --eval
```

### Pretrained Models Zoo

| Model  | Type        | Architecture    | Dataset  | Epochs | Checkpoint                                                                                     |
|--------|-------------|-----------------|----------|--------|------------------------------------------------------------------------------------------------|
| DETReg | Pretraining | Deformable DETR | ImageNet | 5      | [link](https://github.com/amirbar/DETReg/releases/download/1.0.0/checkpoint_imagenet.pth)      |
| DETReg | Pretraining | DETR            | ImageNet | 60     | [link](https://github.com/amirbar/DETReg/releases/download/1.0.0/checkpoint_imagenet_detr.pth) |
| DETReg | Pretraining | Deformable DETR | MSCoco   | 50     | [link](https://github.com/amirbar/DETReg/releases/download/1.0.0/checkpoint_coco.pth)          |
| DETReg | Finetuned   | Deformable DETR | MSCoco   | 50     | [link](https://github.com/amirbar/DETReg/releases/download/1.0.0/full_coco_finetune.pth)       |

## Citation
If you found this code helpful, feel free to cite our work: 

```bibtext
@misc{bar2021detreg,
      title={DETReg: Unsupervised Pretraining with Region Priors for Object Detection},
      author={Amir Bar and Xin Wang and Vadim Kantorov and Colorado J Reed and Roei Herzig and Gal Chechik and Anna Rohrbach and Trevor Darrell and Amir Globerson},
      year={2021},
      eprint={2106.04550},
      archivePrefix={arXiv},
      primaryClass={cs.CV}
}
```

## Related Works
If you found DETReg useful, consider checking out these related works as well: [ReSim](https://github.com/Tete-Xiao/ReSim), [SwAV](https://github.com/facebookresearch/swav), [DETR](https://github.com/facebookresearch/detr), [UP-DETR](https://github.com/dddzg/up-detr), and [Deformable DETR](https://github.com/fundamentalvision/Deformable-DETR).

## Change Log
* 04/28/22 - Bug fix in multiprocessing, up-to-date results [here](docs/full-semi-sup.png) 
* 12/13/21 - Add DETR architecture
* 12/12/21 - Update experiments hyperparams in accordance with new paper version
* 12/12/21 - Avoid box caching on TopK policy (bug fix)
* 9/19/21 - Fixed Pascal VOC training with %X of training data


## Acknowlegments
DETReg builds on previous works code base such as [Deformable DETR](https://github.com/fundamentalvision/Deformable-DETR) and [UP-DETR](https://github.com/dddzg/up-detr). If you found DETReg useful please consider citing these works as well.

## License
DETReg is released under the Apache 2.0 license. Please see the [LICENSE](https://github.com/amirbar/DETReg/blob/main/LICENSE) file for more information.
