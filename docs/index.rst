.. vodet documentation master file, created by
   sphinx-quickstart on Tue Feb  2 17:31:07 2021.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

Welcome to vodet's documentation!
=================================

Variational Object DETector for time-lapse images.

:code:`vodet` is a simple library for semi-supervised object detection for 
anything using `pixyz <https://github.com/masa-su/pixyz>`_ and 
`PyTorch <https://pytorch.org/>`_. 
Since `Gaussian-Mixture Variational Auto Encoder 
(GMVAE) <http://ruishu.io/2016/12/25/gmvae/>`_ is used in the 
image classifier, it requires a very small amount (two or three) of 
annotation images. Though `vodet` assumes time-lapse images, 
it may work well with other kinds of images if the sizes of objects 
do not vary by images.


.. toctree::
   :maxdepth: 2
   :caption: Contents:
   
   installation
   algorithm
   usage
   vodet
