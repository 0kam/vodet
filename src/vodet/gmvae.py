from torch import nn
import torch_optimizer as optim
import torch
from torch._C import import_ir_module
from torch.nn import functional as F
from  torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from torchvision import datasets
from tensorboardX import SummaryWriter

import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import datetime
import copy
from glob import glob
from pathlib import Path

import pixyz
from pixyz.losses import ELBO
from pixyz.models import Model

from vodet.utils import set_patches, plot_latent, plot_reconstruction
from vodet.distributions import Generator, Inference, Classifier, Prior
from vodet.detect import Detector

class GMVAE:
    """
    GMVAE object for object detection.
    """
    def __init__(self, data_dirs):
        """
        Parameters
        ----------
        data_dirs : dict
            A dict that contains
            - "train":path_for_train_directory
            - "validation":path_for_validation_directory
            - "unlabelled":path_for_unlabelled_directory
            All directories must have "source" subdirectory that contains source images.
            The "train" and "validation" directory must also have "labels" directory that contains label data files (VoTT csv export or labelme json output)
        """
        self.data_dirs = data_dirs

    def set_patches(self, label_type, step_ratio=1.0):
        """
        Split source images into patches with labels to train GMVAE classifier.

        Parameters
        ----------
        label_type : str
            The type of label images. Either "VoTT" for VoTT's csv export or "labelme" for labelme's json output.
        step_ratio : float default 1.0
            Sliding window step size relative to the size of patches.
        """
        set_patches(self.data_dirs, label_type, step_ratio)
    
    def set_dataloaders(self, batch_size, transforms):
        """
        Set up dataloaders for training.

        Parameters
        ----------
        batch_size : int
            The batch size of dataloaders.
        transforms: dict 
            A dict of transforms each made by torchvision.transforms.Comose().
            The keys must be "train", "validation" and "unlabelled"
            At least, `transforms.Resize((24,24))` and `transforms.ToTensor()` is required.
            The size of `transforms.Resize()` **must be (24,24)**.

        """
        unlabelled_ds = datasets.ImageFolder(self.data_dirs["unlabelled"]+"/patches", transforms["unlabelled"])
        train_ds = datasets.ImageFolder(self.data_dirs["train"]+"/patches", transforms["labelled"])
        val_ds = datasets.ImageFolder(self.data_dirs["validation"]+"/patches", transforms["validation"])
        self.unlabelled = DataLoader(unlabelled_ds, batch_size, shuffle=True)
        self.labelled = DataLoader(train_ds, batch_size, shuffle=True)
        self.validation = DataLoader(val_ds, batch_size, shuffle=True)
        self.classes = train_ds.class_to_idx
        self.y_dim = len(self.classes)
    
    def set_model(self, z_dim, device):
        """
        Set up model for training.
        
        Parameters
        ----------
        z_dim : int
            Dimension of the latent variable.
        device : str
            The name of device for training.
        """
        # distributions for supervised learning
        self.p = Generator(z_dim).to(device)
        self.q = Inference(z_dim, self.y_dim).to(device)
        self.f = Classifier(self.y_dim).to(device)
        self.prior = Prior(z_dim, self.y_dim).to(device)
        self.p_joint = self.p * self.prior
        self.device = device
        self.z_dim = z_dim

        # distributions for unsupervised learning
        _q_u = self.q.replace_var(x="x_u", y="y_u")
        p_u = self.p.replace_var(x="x_u")
        f_u = self.f.replace_var(x="x_u", y="y_u")
        prior_u = self.prior.replace_var(y="y_u")
        q_u = _q_u * f_u
        p_joint_u = p_u * prior_u
        
        p_joint_u.to(device)
        q_u.to(device)
        f_u.to(device)
        
        elbo_u = ELBO(p_joint_u, q_u)
        elbo = ELBO(self.p_joint, self.q)
        nll = -self.f.log_prob() # or -LogProb(f)
        
        rate = 1 * (len(self.unlabelled) + len(self.labelled)) / len(self.labelled)
        
        self.loss_cls = -elbo_u.mean() -elbo.mean() + (rate * nll).mean()
        self.test_loss_cls = nll.mean() #-elbo.mean()
        self.model = Model(self.loss_cls,test_loss=self.test_loss_cls,
                      distributions=[self.p, self.q, self.f, self.prior], optimizer=optim.RAdam, optimizer_params={"lr":1e-3})
        print("Model:")
        print(self.model)
    
    def _train(self, epoch):
        train_loss = 0
        labelled_iter = self.labelled.__iter__()
        for x_u, y_u in tqdm(self.unlabelled):
            try: 
                x, y = labelled_iter.next()
            except StopIteration:
                labelled_iter = self.labelled.__iter__()
                x, y = labelled_iter.next()
            x = x.to(self.device)
            y = torch.eye(self.y_dim)[y].to(self.device)
            x_u = x_u.to(self.device)
            loss = self.model.train({"x": x, "y": y, "x_u": x_u})
            train_loss += loss
        train_loss = train_loss * self.unlabelled.batch_size / len(self.unlabelled.dataset)
        print('Epoch: {} Train loss: {:.4f}'.format(epoch, train_loss))
        return train_loss
    
    def _test(self, epoch):
        test_loss = 0
        total = [0 for _ in range(len(self.classes))]
        tp = [0 for _ in range(len(self.classes))]
        fp = [0 for _ in range(len(self.classes))]
        for x, _y in self.validation:
            x = x.to(self.device)
            y = torch.eye(self.y_dim)[_y].to(self.device)
            loss = self.model.test({"x": x, "y": y})
            test_loss += loss
            pred_y = self.f.sample_mean({"x": x}).argmax(dim=1)
            for c in list(self.classes.values()):
                pred_yc = pred_y[_y==c]
                _yc = _y[pred_y==c]
                total[c] += len(_y[_y==c])
                tp[c] += len(pred_yc[pred_yc==c])
                fp[c] += len(_yc[_yc!=c])
        
        test_loss = test_loss * self.validation.batch_size / len(self.validation.dataset)
        test_recall = [100 * c / t for c,t in zip(tp, total)]
        test_precision = []
        for _tp,_fp in zip(tp, fp):
            if _tp + _fp == 0:
                test_precision.append(0)
            else:
                test_precision.append(100 * _tp / (_tp + _fp))
        recall = self.classes.copy()
        prec = self.classes.copy()
        i = 0
        for c in recall:
            recall[c] = test_recall[i]
            prec[c] = test_precision[i]
            i += 1
        print("Test Loss:", str(test_loss), "Test Recall:", str(recall), "Test Precision:", str(prec))
        return test_loss, recall, prec
    
    def train(self, epochs, precision_th=90):
        """
        Train model.

        Parameters
        ----------
        epochs : int
            Epochs to train.
        precision_th : float
            Precision threshold (percent). 
            If the minimum precision in each epoch is larger than this value and the test loss is lower than previous "best model", the "best model" will be overwritten.
        """
        dt_now = datetime.datetime.now()
        exp_time = dt_now.strftime('%Y%m%d_%H_%M_%S')
        writer = SummaryWriter(str(Path(self.data_dirs["train"]).parent) + \
            "/runs/" + "vodet_gmvae_" + exp_time)
        
        _x = []
        _y = []
        for i in range(10):
            _xx, _yy = iter(self.validation).next()
            _x.append(_xx)
            _y.append(_yy)
        
        _x = torch.cat(_x, dim = 0)
        _y = torch.cat(_y, dim = 0)
        
        _x = _x.to(self.device)
        _y = torch.eye(self.y_dim)[_y].to(self.device)
        
        # for plot reconstruction
        _xr = []
        _yr = []
        for i in range(self.y_dim):
            _y_recon = _y.argmax(dim = 1)
            _x_recon = _x[_y_recon==i,:,:,:]
            _y_recon = _y[_y_recon==i,:]
            _xr.append(_x_recon[0:8,:,:,:])
            _yr.append(_y_recon[0:8,:])
        
        _xr = torch.cat(_xr, dim = 0)
        _yr = torch.cat(_yr, dim = 0)
        
        best_test_loss = 10000
        for epoch in range(1, epochs + 1):
            train_loss = self._train(epoch)
            test_loss, test_recall, test_precision = self._test(epoch)
            if test_loss < best_test_loss and float(min(test_precision.values())) >= float(precision_th):
                self.best_p = Generator(self.z_dim).to(self.device)
                self.best_q = Inference(self.z_dim, self.y_dim).to(self.device)
                self.best_f = Classifier(self.y_dim).to(self.device)
                self.best_p.load_state_dict(self.p.state_dict())
                self.best_q.load_state_dict(self.q.state_dict())
                self.best_f.load_state_dict(self.f.state_dict())
                self.best_recall = test_recall
                self.best_precision = test_precision
            writer.add_scalar('train_loss', train_loss.item(), epoch)
            writer.add_scalar('test_loss', test_loss.item(), epoch)
            for label in test_recall:
                writer.add_scalar("test_recall_" + label, test_recall[label], epoch)
                writer.add_scalar("test_precision_" + label, test_precision[label], epoch)
            # reconstructed images
            recon = plot_reconstruction(_xr, _yr, self.p, self.q)
            latent = plot_latent(_x, _y, self.f, self.q, self.y_dim)
            writer.add_images("Image_reconstruction", recon, epoch)
            writer.add_images("Image_latent", latent, epoch)
        
        writer.close()
    
    def detector(self, label_type = "labelme", conf_th = 0.95, iou_th = 0.3, step_ratio = 0.5, input_size = [24, 24]):
        """
        Create a Detector object.

        Parameters
        ----------
        label_type : str
            The type of label data, either "VoTT" (for VoTT's csv export) or "labelme" (for labelme's json export).
        conf_th : float default 0.99
            The confidence threshold for each proposed bounding box.
        iou_th : float default 0.3
            The threshold of IoU value for Non-Maximum Supression of bounding boxes.
        step_ratio : float default 0.5
            The ratio between the step size and width or height of sliding windows.
        input_size : list of int
            The input size of the classifier
        
        Returns
        -------
        d : vodet.detect.Detector
            A Detector instance.
        """
        if label_type == "VoTT":
            labels_path = glob(self.data_dirs["train"] + "/labels/*.csv")[0]
        elif label_type == "labelme":
            labels_path = self.data_dirs["train"] + "/labels/"
        else:
            raise Exception("Error: the annotation data is not supported")
        d = Detector(self.best_f, self.classes, labels_path, label_type, conf_th , iou_th, step_ratio, input_size)
        return d
