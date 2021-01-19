from glob import glob
from PIL import Image, ImageDraw
import torch
import numpy as np
import pandas as pd
import statistics as stat
import math
from pyclustering.cluster.xmeans import xmeans
from pyclustering.cluster.center_initializer import kmeans_plusplus_initializer
from tqdm import tqdm
from glob import glob
from pathlib import Path
from matplotlib import pyplot as plt
import os
import shutil

from set_images import read_labelme
from utils import discrete_cmap

# bounding box selection 
## Intersection over Union
def iou(a: tuple, b: tuple) -> float:
    a_x1, a_y1, a_x2, a_y2 = a
    b_x1, b_y1, b_x2, b_y2 = b
    
    if a == b:
        return 1.0
    elif (
        (a_x1 <= b_x1 and a_x2 > b_x1) or (a_x1 >= b_x1 and b_x2 > a_x1)
    ) and (
        (a_y1 <= b_y1 and a_y2 > b_y1) or (a_y1 >= b_y1 and b_y2 > a_y1)
    ):
        intersection = (min(a_x2, b_x2) - max(a_x1, b_x1)) * (min(a_y2, b_y2) - max(a_y1, b_y1))
        union = (a_x2 - a_x1) * (a_y2 - a_y1) + (b_x2 - b_x1) * (b_y2 - b_y1) - intersection
        return intersection / union
    else:
        return 0.0

## Non-Maximum Suppression
def nms(bboxes: list, scores: list, labels: list, iou_threshold: float) -> (list, list):
    new_bboxes = []
    new_labels = []
    
    while len(bboxes) > 0:
        i = scores.index(max(scores))
        bbox = bboxes.pop(i)
        scores.pop(i)
        label = labels.pop(i)
        
        deletes = []
        for j, (bbox_j, score_j) in enumerate(zip(bboxes, scores)):
            if iou(bbox, bbox_j) > iou_threshold:
                deletes.append(j)
                
        for j in deletes[::-1]:
            bboxes.pop(j)
            scores.pop(j)
                
        new_bboxes.append(bbox)
        new_labels.append(label)
        
    return new_bboxes, new_labels

def soft_nms(bboxes: list, scores: list, labels:list, iou_threshold: float) -> (list, list, list):
    new_bboxes, new_scores, new_labels = [], [], []
    
    while len(bboxes) > 0:
        i = scores.index(max(scores))
        bbox = bboxes.pop(i)
        score = scores.pop(i)
        label = labels.pop(i)
        
        for j, (bbox_j, score_j) in enumerate(zip(bboxes, scores)):
            iou_j = iou(bbox, bbox_j)
            if iou_j > iou_threshold:
                scores[j] = score_j * (1 - iou_j)
                
        new_bboxes.append(bbox)
        new_scores.append(score)
        new_labels.append(label)

        
    return new_bboxes, new_scores, new_labels

def detect(classifier, image_path, labels_path, out_path, classes, label_type, bb_color, conf_th=0.5, iou_th=0.3, step_ratio = 0.5, input_size=[24,24], device="cuda:0"):
    if label_type == "VoTT":
        label_data = pd.read_csv(labels_path)
    elif label_type == "labelme":
        label_data = read_labelme(labels_path)
    else:
        raise Exception("Error: the annotation data is not supported")
    labels = list(label_data["label"].unique())
    label_data = label_data.assign(x = lambda df: df.xmax-df.xmin)
    label_data = label_data.assign(y = lambda df: df.ymax-df.ymin)
    label_data[["x", "y"]] = label_data[["x", "y"]].astype("int")
    
    x_bound = math.ceil(max(label_data["x"]) / 2)
    y_bound = math.ceil(max(label_data["y"]) / 2)
    
    img = Image.open(image_path)
    
    xy = label_data[["x", "y"]].values
    
    xm_c = kmeans_plusplus_initializer(xy, 4).initialize()
    xm_i = xmeans(data=xy, initial_centers=xm_c, kmax=10, ccore=True)
    xm_i.process()
    xy = xm_i._xmeans__centers
    
    df = pd.DataFrame({"label":[],
                       "conf":[],
                       "xmin":[],
                       "ymin":[],
                       "xmax":[],
                       "ymax":[]})
    i = j = 0
    
    for size in tqdm(xy):
        step = [int(size[0]*step_ratio), int(size[1]*step_ratio)]
        x_centers = range(x_bound, img.size[0]-x_bound, step[0])
        y_centers = range(y_bound, img.size[1]-y_bound, step[1])
        for x in x_centers:
            for y in y_centers:
                box = (int(x-size[0]/2), int(y-size[1]/2), int(x+size[0]/2), \
                    int(y+size[1]/2))
                patch = img.crop(box)
                patch = patch.resize(input_size)
                patch = np.array(patch) / 255
                patch = np.transpose(patch, (2, 0, 1))
                patch = torch.from_numpy(patch)
                patch = patch.float().unsqueeze(0).to(device)
                with torch.no_grad():
                    classifier.eval()
                    _y_pred = classifier.sample_mean({"x":patch}).detach().cpu()
                    y_pred = _y_pred.argmax(1)
                    conf = _y_pred.max()
                    label = [k for k, v in classes.items() if v == int(y_pred)]
                    df = df.append({"label":label[0],
                              "conf":conf.item(),
                              "xmin":box[0],
                              "ymin":box[1],
                              "xmax":box[2],
                              "ymax":box[3]},
                              ignore_index=True)
        
                j += 1
            i += 1
    df = df[df["label"]!="other"] \
        [df["conf"] > conf_th]
    
    bboxes = [(d["xmin"],d["ymin"],d["xmax"],d["ymax"]) for _, d in df.iterrows()]
    confs = [d["conf"] for _, d in df.iterrows()]
    labels = [d["label"] for _, d in df.iterrows()]
    nms_res = nms(bboxes, confs, labels, iou_th)
    
    df2 = pd.DataFrame({"label":[],
                       "xmin":[],
                       "ymin":[],
                       "xmax":[],
                       "ymax":[]})
    
    for r in range(len(nms_res[0])):
        df2 = df2.append({"label":nms_res[1][r],
                        "xmin":nms_res[0][r][0],
                        "ymin":nms_res[0][r][1],
                        "xmax":nms_res[0][r][2],
                        "ymax":nms_res[0][r][3]}, ignore_index=True)
    
    for _, d in df2.iterrows():
        draw = ImageDraw.Draw(img)
        text_w, text_h = draw.textsize(d["label"])
        label_y = d["ymin"] if d["ymin"] <= text_h else d["ymin"] - text_h
        draw.rectangle((d["xmin"], label_y, d["xmax"], d["ymax"]), outline=bb_color[d["label"]])
        draw.rectangle((d["xmin"], label_y, d["xmin"]+text_w, label_y+text_h), outline=bb_color[d["label"]], fill=bb_color[d["label"]])
        draw.text((d["xmin"], label_y), d["label"], fill=(255,255,255))
    
    detected_numbers = {}
    
    for c in classes:
        if c != "other":
            d = df2[df2["label"]==c]
            detected_numbers[c] = len(d)
    
    img.save(out_path)
    return detected_numbers

class Detector:
    def __init__(self, classifier, classes, labels_path, label_type = "labelme", conf_th = 0.95, iou_th = 0.3, step_ratio = 0.5, input_size = [24, 24], device="cuda:0"):
        self.classifier = classifier
        self.classes = classes
        self.labels_path = labels_path
        self.label_type = label_type
        self.conf_th = conf_th
        self.iou_th = iou_th
        self.step_ratio = step_ratio
        self.input_size = input_size
        self.device = device
    
    def detect_img(self, in_path, out_path):
        cmap = discrete_cmap(len(self.classes), "hsv")
        bb_color = {}
        i = 0
        for c in self.classes:
            if c != "other":
                bb_color[c] = tuple(int(cmap(i)[j]*255) for j in range(3))
            i += 1
        nums = detect(self.classifier, in_path, self.labels_path, out_path, self.classes, self.label_type, bb_color, self.conf_th, self.iou_th, self.step_ratio, self.input_size, self.device)
        return nums
    
    def detect_dir(self, in_dir, out_dir):
        if os.path.exists(out_dir):
            shutil.rmtree(out_dir)
        os.mkdir(out_dir)

        columns = list(self.classes.keys())
        columns.append("image")
        columns.remove("other")
        df = pd.DataFrame(columns=columns)
        files = glob(in_dir+"/*")
        for f in files:
            num = self.detect_img(f, f.replace(in_dir, out_dir))
            num["image"] = Path(f).name
            df = df.append(num, ignore_index=True)
        self.detected = df
        return df
    
    def draw_barplot(self, date_df, out_path):
        df = pd.merge(self.detected, date_df)
        df = df.set_index("date").sort_index()
        self.detected = df
        labels = list(self.classes.keys())
        labels.remove("other")
        df.plot.bar(y=labels, rot = "45", figsize=(6, 4))
        plt.subplots_adjust(bottom=0.2)
        plt.savefig(out_path)
        return df
        
