from os import error
from PIL import Image
from PIL.ExifTags import TAGS
from glob import glob
from pathlib import Path
import re
import numpy as np
import pandas as pd
import statistics as stat
import math
import os
import shutil
import json
from tqdm import tqdm
import random
from matplotlib import pyplot as plt
import torch
import datetime

def read_labelme(in_dir):
    """
    Read labelme's json files and convert it into pd.DataFrame

    Parameters
    ----------
    in_dir : str
        The directory where json files are stored.

    Returns
    -------
    dataframe : pd.DataFrame
    """
    files = glob(in_dir + "/*.json")
    dataframe = pd.DataFrame({"image":[], "label":[], "xmin":[], "ymin":[], "xmax":[], "ymax":[]})
    for f in files:
        data = json.load(open(f))
        df = pd.DataFrame(data["shapes"])
        df = df.assign(xmin = [min(p[0][0], p[1][0]) for p in df.points]). \
            assign(ymin = [min(p[0][1], p[1][1]) for p in df.points]). \
            assign(xmax = [max(p[0][0], p[1][0]) for p in df.points]). \
            assign(ymax = [max(p[0][1], p[1][1]) for p in df.points]). \
            assign(image = Path(data["imagePath"]).name). \
            filter(["image", "label", "xmin", "ymin", "xmax", "ymax"])
        dataframe = pd.merge(dataframe, df, how = "outer")
    return dataframe

def make_patches_labelled(data_dir, out_dir, label_data, step_ratio):
    """
    Split source images with labels.

    Parameters
    ----------
    data_dir : str
        The path for data directory.
    out_dir : str
        The path for output (patched images) directory.
    step_ratio : float
        Sliding window step size relative to the size of patches.
    """
    img_paths = [str(p) for p in glob(data_dir + "/source/*") \
        if re.search(".*\.[jpg, jpeg, JPG, png]", p)] 
    if os.path.exists(out_dir):
        shutil.rmtree(out_dir)
    os.mkdir(out_dir)
    labels = list(label_data["label"].unique())
    labels.append("other")
    for label in labels:
        os.mkdir(out_dir + "/" + label)
    
    for p in tqdm(img_paths):
        img = Image.open(p)
        name = Path(p).name
        ls = label_data[label_data["image"] == name]
        mask = np.zeros([img.size[0], img.size[1]])
        for i in range(len(ls)):
            l = ls.iloc[i:i+1,:]
            box = (int(l["xmin"]),int(l["ymin"]),int(l["xmax"]),int(l["ymax"]))
            label = l["label"].values[0]
            fn = out_dir + "/" + label + "/" + name.replace(Path(name).suffix, "") + "_" + str(i) + ".png"
            croped = img.crop(box)
            croped.save(fn)
            # mask
            mask[box[0]:box[2], box[1]:box[3]] = 1
        
        x_mean = int(stat.mean(label_data["x"]))
        y_mean = int(stat.mean(label_data["y"]))
        x_bound = math.ceil(max(label_data["x"]))
        y_bound = math.ceil(max(label_data["y"]))
        step = [int(x_mean * step_ratio), int(y_mean * step_ratio)]

        x_centers = range(x_bound, img.size[0]-x_bound, step[0])
        y_centers = range(y_bound, img.size[1]-y_bound, step[1])
        i = j = 0
        for x in x_centers:
            for y in y_centers:
                size = label_data[["x","y"]].sample(1).values
                box = (int(x-size[0,0]/2), int(y-size[0,1]/2), int(x+size[0,0]/2), \
                    int(y+size[0,1]/2))
                m = mask[int(box[0]+(size[0,0]/4)):int(box[2]-(size[0,0]/4)),\
                     int(box[1]+(size[0,1]/4)):int(box[3]-size[0,1]/4)]
                if m.max() < 0.5: # if the central area of this patch does not contain any labels
                    croped = img.crop(box)
                    fn = out_dir + "/other/" + name.replace(Path(name).suffix, "") + "_" + str(i) + "_" + str(j) + ".png"
                    croped.save(fn)
                    j += 1
            i += 1


def make_patches_unlabelled(data_dir, out_dir, label_data):
    """
    Split source images into patches. The patch sizes are determined by label_data.

    Parameters
    ----------
    data_dir : str
        The path for data directory.
    out_dir : str
        The path for output (patched images) directory.
    label_data : pd.DataFrame
        A dataframe of label data. This will be used to determine the sizes of patches.
    step_ratio : float
        Sliding window step size relative to the size of patches.
    """
    img_paths = [str(p) for p in glob(data_dir + "/source/*") \
        if re.search(".*\.[jpg, jpeg, JPG, png]", p)] 
    if os.path.exists(out_dir):
        shutil.rmtree(out_dir)
    os.mkdir(out_dir)
    labels = list(label_data["label"].unique())
    labels.append("other")
    os.mkdir(out_dir + "/unlabelled")
    
    for p in tqdm(img_paths):
        img = Image.open(p)
        name = Path(p).name
        
        x_mean = int(stat.mean(label_data["x"]))
        y_mean = int(stat.mean(label_data["y"]))
        x_bound = math.ceil(max(label_data["x"]))
        y_bound = math.ceil(max(label_data["y"]))
        step = [int(x_mean * step_ratio), int(y_mean * step_ratio)]

        x_centers = range(x_bound, img.size[0]-x_bound, step[0])
        y_centers = range(y_bound, img.size[1]-y_bound, step[1])
        i = j = 0
        for x in x_centers:
            for y in y_centers:
                size = label_data[["x","y"]].sample(1).values
                box = (int(x-size[0,0]/2), int(y-size[0,1]/2), int(x+size[0,0]/2), \
                    int(y+size[0,1]/2))
                croped = img.crop(box)
                fn = out_dir + "/unlabelled/" + name.replace(Path(name).suffix, "") + "_" + str(i) + "_" + str(j) + ".png"
                croped.save(fn)
                j += 1
            i += 1

def set_patches(data_dirs, label_type, step_ratio=1.0):
    """
    Making labelled and unlabelled patches from annotated images. 
    The annotation shape must be rectangulars.
    Only compatible with VoTT (csv export) and labellme.
    
    Parameters
    ----------
    data_dirs: dict
        A dictionary with 3 components, "train", "validation" and "unlabelled" . 
        The "train" and "validation" directory should have
        - /source: a sub-directory with source images
        - /labels: a sub-directory with label data. label data should be VoTT csv-export file (.csv) or labelme output file (.json)
    label_type: str
        The type of label files, "VoTT" and "labelme" are available.
    step_ratio : float default 1.0
        Sliding window step size relative to the size of patches.
    """  
    # train data
    ## label set up
    if label_type == "VoTT":
        label_data = pd.read_csv(glob(data_dirs["train"] + "/labels/*.csv")[0])
    elif label_type == "labelme":
        label_data = read_labelme(data_dirs["train"] + "/labels/")
    else:
        raise Exception("Error: the annotation data is not supported")

    labels = list(label_data["label"].unique())
    labels.append("other")
    label_data = label_data.assign(x = lambda df: df.xmax-df.xmin)
    label_data = label_data.assign(y = lambda df: df.ymax-df.ymin)
    label_data[["x", "y"]] = label_data[["x", "y"]].astype("int")
    ## generate patches
    print("Start processing train data!")
    make_patches_labelled(data_dirs["train"], data_dirs["train"]+"/patches", label_data, step_ratio)

    # validation data
    ## label set up
    if label_type == "VoTT":
        label_data = pd.read_csv(glob(data_dirs["validation"] + "/labels/*.csv")[0])
    elif label_type == "labelme":
        label_data = read_labelme(data_dirs["validation"] + "/labels/")
    else:
        raise Exception("Error: the annotation data is not supported")

    labels = list(label_data["label"].unique())
    labels.append("other")
    label_data = label_data.assign(x = lambda df: df.xmax-df.xmin)
    label_data = label_data.assign(y = lambda df: df.ymax-df.ymin)
    label_data[["x", "y"]] = label_data[["x", "y"]].astype("int")
    ## generate patches
    print("Start processing validation data!")
    make_patches_labelled(data_dirs["validation"], data_dirs["validation"]+"/patches", label_data, step_ratio)

    # unlabelled data
    ## to define the patch size, use label data for train
    if label_type == "VoTT":
        label_data = pd.read_csv(glob(data_dirs["train"] + "/labels/*.csv")[0])
    elif label_type == "labelme":
        label_data = read_labelme(data_dirs["train"] + "/labels/")
    else:
        raise Exception("Error: the annotation data is not supported")

    labels = list(label_data["label"].unique())
    labels.append("other")
    label_data = label_data.assign(x = lambda df: df.xmax-df.xmin)
    label_data = label_data.assign(y = lambda df: df.ymax-df.ymin)
    label_data[["x", "y"]] = label_data[["x", "y"]].astype("int")
    ## generate patches
    print("Start processing unlabelled data!")
    make_patches_unlabelled(data_dirs["unlabelled"], data_dirs["unlabelled"]+"/patches", label_data)

def plot_reconstruction(x, y, p, q):
    with torch.no_grad():
        z = q.sample({"x":x, "y":y}, return_all=False)
        recon_batch = p.sample_mean(z)
        recon = torch.cat([x, recon_batch]).cpu()
        return recon

# borrowed from https://gist.github.com/jakevdp/91077b0cae40f8f8244a
def discrete_cmap(N, base_cmap=None):
    """Create an N-bin discrete colormap from the specified input map"""

    # Note that if base_cmap is a string or None, you can simply do
    #    return plt.cm.get_cmap(base_cmap, N)
    # The following works for string, None, or a colormap instance:

    base = plt.cm.get_cmap(base_cmap)
    color_list = base(np.linspace(0, 1, N))
    cmap_name = base.name + str(N)
    return base.from_list(cmap_name, color_list, N)

# borrowed from https://github.com/dragen1860/pytorch-mnist-vae/blob/master/plot_utils.py
def plot_latent(x, y, f, q, y_dim):
    with torch.no_grad():
        label = torch.argmax(y, dim = 1).detach().cpu().numpy()
        _y = f.sample_mean({"x":x})
        z = q.sample_mean({"x":x, "y":_y}).detach().cpu().numpy()
        N = y_dim
        fig = plt.figure(figsize=(8, 6))
        plt.scatter(z[:, 0], z[:, 1], c=label, marker='o', edgecolor='none', cmap=discrete_cmap(N, 'jet'))
        plt.colorbar(ticks=range(N))
        plt.grid(True)
        fig.canvas.draw()
        image = fig.canvas.renderer._renderer
        image = np.array(image).transpose(2, 0, 1)
        image = np.expand_dims(image, 0)
        return image


def exif_date(in_dir):
    """
    Read Exif data of images in a directory and returns a dataframe of file names of images and their shooting date.

    Parameters
    ----------
    in_dir : str
        Path for the directory of images.

    Returns
    -------
    df : pd.DataFrame
        A dataframe of filename and shooting date.         
    """
    files = [str(p) for p in glob(in_dir + "/**/*", recursive=True) \
        if re.search(".*\.[jpg, jpeg, JPG, png]", p)] 
    
    df = pd.DataFrame({
        "image":[],
        "date":[]
    })
    for f in files:
        im = Image.open(f)
        try:
            exif = im._getexif()
        except AttributeError:
            exif = None
        
        exif_table = {}
        if type(exif) == dict:
            for tag_id, value in exif.items():
                tag = TAGS.get(tag_id, tag_id)
                exif_table[tag] = value  
            date = exif_table["DateTime"]
            dt=datetime.datetime.strptime(date, "%Y:%m:%d %H:%M:%S")
            df = df.append({
                "image":Path(f).name,
                "date":dt.date()
            }, ignore_index = True)
    
    return df