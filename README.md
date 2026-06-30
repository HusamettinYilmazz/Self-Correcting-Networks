<div align="center">
  <img src="assets/readme_images/training_arch.png" alt="Background Image" width="95%" />
</div>

<h1 align="center"> Semi-Supervised Semantic Image Segmentation with Self-correcting Networks </h1>

<p align="center">
  A pytorch implementation of the <strong>CVPR 2020 paper</strong>, <a href="https://openaccess.thecvf.com/content_CVPR_2020/papers/Ibrahim_Semi-Supervised_Semantic_Image_Segmentation_With_Self-Correcting_Networks_CVPR_2020_paper.pdf"><em>Semi-Supervised Semantic Image Segmentation with Self-correcting Networks</em></a> by <strong>Ibrahim et al.</strong>
  This framework enables joint learning from a small fully supervised dataset and large weakly labeled data, using self-correcting networks to refine predictions and reduce annotation noise 
</p>


## Table of Content
1. [Stage 1: Ancillary Model Training](#stage-1-ancillary-model-training)
    - [Model Architecture](#model-architecture)
    - [Loss Function](#loss-function)
    - [Hyperparameters](#hyperparameteres)
    - [Training Setup](#training-setup)
    - [Training Dataset](#training-dataset)
    - [Ancillary Model mIoU](#ancillary-model-miou)
2. [Stage 2 Correcting Model Training](#stage-2-correcting-model-training)
    - [Model Architecture](#model-architecture)
    - [Loss Function](#loss-function)
    - [Hyperparameters](#hyperparameteres)
    - [Training Setup](#training-setup)
    - [Training Dataset](#training-dataset)
    - [Ancillary Model mIoU](#ancillary-model-miou)


<strong>The approach in the paper divides training into three stages: the first trains the ancillary model, the second trains the self-correcting network, and the third focuses on the primary model.</strong>

## Stage 1 Ancillary Model Training

<div align="center">
  <img src="assets/readme_images/ancillary_model_arch.png" alt="Background Image" width="95%" />
</div>
Stage 1 trains the ancillary model to learn robust representations from weakly labeled examples.

### Model Architecture
the model is built on standard encoder-decoder segmentation models (DeepLabV3+ is used in my implementation), the paper is extending the architecture with additional bounding box input.

1. Image Encoder:
The input image is processed by a pretrained conv encoder (Resnet101 is used in my implementation). it produces 2 scale feature maps (low and high feature maps).

2. Bounding Box Encoder:
This encoder takes the weak labeled mask to encode box information into spatial attention map.

The input weak mask is resized (using 3x3 Conv followed by sigmoid activation) to match the spatial resolution of low and high feature map.

3. Feature Attention Fusion:
In this step the output of bounding box encoder (box low and high) are fused with the low and high attention map from Image encoder using <strong>Element wise multiplication </strong>.

4. Decoder:
Same as Deeplabv3+ decoder the decoder is expecting multi scale feature maps.
Both fused low and high scales are passed to decoder 
low is passed to internal layer of the decoder and
high is passed to the begining of the decoder.

### Loss Function
The model is learning conditional probability distribution over input image (x) and weak labeled mask (b)
<p align="center">P<sub>anc</sub>(y | x, b)</p>

The loss function is normal <strong>Cross Entropy</strong>
<p align="center">
L = -log P<sub>anc</sub>(y | x, b)
</p>

### Hyperparameters
|Parameter|Value|
|---|---|
|Learning rate| 7e-3 |
|Batch size| 4 each GPU |
|# GPUs| 2 |
|Scheduler|custom: step based scheduler|
|Optimizer| SGD |
|Gradient Accumulation| 1 |
|Stage1 Epochs| 184 |
|weight_decay| 4e-5 |

### Training Setup
The paper authours used <strong>4 GPUs, each with a batch of 4 images</strong>.
I trained the model on Kaggle free tier <strong>2 GPU T4 each with a batch of 4 images.</strong> kaggle doesn't allow continous training more than 12 hours. so I trained multible iterations on total it took around 48 hours.

### Training Dataset
Half of the fully supervised training dataset is used in this stage to prevent data leakage in the next stage where the self-correcting network is using the ancillary model outputs.

### Ancillary Model mIoU
I reached ~82.8 mIoU while paper reports ~85.5 the difference may come from multiple reasons
1. My total batch size is 8 when paper use batch size of 16, which results in different values for BatchNorm layers.
2. The output stride value of backbone (Xception65) due to GPU limitations I used 16 If they used 8 this can make a little bit difference in mIoU.
3. The split of fully supervised dataset: The split can contribute to the mIoU if it is better on matching validation dataset distribution.
4. Something I couldn't figure out :)


## Stage 2 Correcting Model Training

<div align="center">
  <img src="assets/readme_images/correcting_model_arch.png" alt="Background Image" width="95%" />
</div>
