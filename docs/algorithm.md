# Algorithm
## Object searching
Because previous object-suggestion methods (e.g. selective search) tend not to be successful for detecting and counting flowers from a photograph of a meadow, which is my purpose to build this module, only traditional brute-force search with given sliding windows has been implemented. The bounding box size is determined by that of train data in the workflow below.
- X-means clustering of  bounding box sizes in training data (initial number of cluster is 4)
- The center of each cluster is determined to be the size of the sliding window
## Label estimation for each patches
After getting a patch with a sliding window, the semi-supervised object classifier runs to estimate its label. This is based on Gaussian Mixture Variational Auto Encoder (GMVAE) proposed by [Rui Shu](http://ruishu.io/2016/12/25/gmvae/). You can see an MNIST experiment of GMVAE [here](https://github.com/0kam/bayesian_dnns/blob/main/README.md).