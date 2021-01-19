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
### Data preparation
