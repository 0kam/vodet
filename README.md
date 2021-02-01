# vodet
Variational Object DETector for time-lapse photographs.

## Overview


## Installation
`pip install git+https://github.com/0kam/vodet`

## Algorithm
### Object searching
Because previous object-sugestion methods (e.g. selective search) tend not to be successful for detecting and counting flowers from a photograph of a meadow, which is my purpose to build this module, only traditional brute-force search with given sliding window has been implemented. The sizes of bounding boxes in train data determine the size of each siding window with the workflow below.
- X-means clustering of sizes of given bounding box (initial number of cluster is 4)
- The centre of each cluster is determined to be the size of sliding window
### Label estimation for each patches
After getting a patch with a sliding window, the semi-supervised object classifier runs to estimate its label. This is based on Gaussian Mixtured Variational Auto Encoder (GMVAE) proposed by [Rui Shu](http://ruishu.io/2016/12/25/gmvae/).

## Usage
### Image Annotating
First, you have to make annotation data to train a classifier.
You should annotate at least two images; one for training and another for validating the model.
The annotation shape have to be rectangulars. Since `vodet` use sliding-window method to detect the objects, very small sizes of rectangulars relative to the size of the image will cause very slow speed of detection. You should also make margines little bit larger than the shilhouette of objects for better classification acuraccy.
You can use either [labelme](https://github.com/wkentaro/labelme) or [VoTT](https://github.com/microsoft/VoTT). Because labelme supports zooming of images, I recommend you to use it especially for high-resolution images.
### Setting up data directories
The data directory should have three subdirectories: `train`, `validation` and `unlabelled`.
The `train` and `validation` directory should have `source` and `labels` subdirectory, that contain source images and label data respectively.
The `unlabelled` directory should only have `source` directory.
The label data should be [VoTT](https://github.com/microsoft/VoTT)'s csv-export (sigle CSV file) or [labelme](https://github.com/wkentaro/labelme)'s json files.
#### Directory structure example
```
.
├── train
│   ├── labels
│   │   └── IMAG0827.json
│   └── source
│       └── IMAG0827.JPG
├── unlabelled
│   └── source
│       ├── IMAG0417.JPG
│       ├── IMAG0441.JPG
│       ├── IMAG0463.JPG
│       ├── IMAG0485.JPG
.               .
.               .
.               .
│       └── IMAG1735.JPG
└── validation
    ├── labels
    │   └── IMAG0986.json
    └── source
        └── IMAG0986.JPG
```

### Creating an GMVAE instance
```
from vodet.gmvae import GMVAE
data_dirs = {
    "train" = path_for_your_train_directory,
    "validation" = path_for_your_validation_directory,
    "unlabelled" = path_for_your_unlabelled_directory
}
gmvae = GMVAE(data_dirs)
```
### Generating patch images
Inorder to train the GMVAE classifier model, first we separate source images into patches with labels based on label data.
With train and validation images, vodet read the label data created by annotation tools. A patch that intersects with rectangular annotations will be labelled as the label name (e.g. `flower`), otherwise `others`. With unlabelled images, sliding-windows crop images into patches. The patch sizes are randomly selelcted from that of train label data. 
```
gmvae.set_patches("labelme") # for labelme
gmvae.set_patches("VoTT") # for VoTT
```
Then `patches` directory will be created inside train, validation, unlabelled directory.
```
.
├── train
│   ├── labels
│   ├── patches
│   │   ├── flower
│   │   └── other
│   └── source
├── unlabelled
 with │   ├── patches
│   │   └── unlabelled
│   └── source
└── validation
    ├── labels
    ├── patches
    │   ├── flower
    │   └── other
    └── source

```
### Preparing Dataloaders
Next, prepare `torch.utils.dataloader` with batch size and `torchvision.transforms`.
```
transforms =
gmvae.set_dataloaders(batch_size=128, transforms=transforms)
```
### Model settings
```

```
