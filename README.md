# vodet
Variational Object DETector for time-lapse images.  
[Documentation](https://vodet.readthedocs.io/en/latest/index.html)

## Overview
`vodet` is a simple library for semi-supervised object detection for anything using [pixyz](https://github.com/masa-su/pixyz) and [PyTorch](https://pytorch.org/). Since [Gaussian-Mixture Variational Auto Encoder (GMVAE)](http://ruishu.io/2016/12/25/gmvae/) is used in the image classifier, it requires a very small amount (two or three) of annotation images. Though `vodet` assumes time-lapse images, it may work well with other kinds of images if the sizes of objects do not vary by images. More details and usages are available at the [documentation](https://vodet.readthedocs.io/en/latest/index.html).

## Installation
`pip install git+https://github.com/0kam/vodet`
## Dependencies
```
pixyz>=0.3.1
torch
pyclustering>=0.10.1.2
matplotlib>=3.1.2
tqdm>=4.50.0
pandas>=1.1.2
torchvision
numpy>=1.17.4
Pillow>=8.1.0
tensorboardX>=2.1
```
## Algorithm
### Object searching
Because previous object-suggestion methods (e.g. selective search) tend not to be successful for detecting and counting flowers from a photograph of a meadow, which is my purpose to build this module, only traditional brute-force search with given sliding windows has been implemented. The bounding box size is determined by that of train data in the workflow below.
- X-means clustering of  bounding box sizes in training data (initial number of cluster is 4)
- The center of each cluster is determined to be the size of the sliding window
### Label estimation for each patches
After getting a patch with a sliding window, the semi-supervised object classifier runs to estimate its label. This is based on Gaussian Mixture Variational Auto Encoder (GMVAE) proposed by [Rui Shu](http://ruishu.io/2016/12/25/gmvae/). You can see an MNIST experiment of GMVAE [here](https://github.com/0kam/bayesian_dnns/blob/main/README.md).
