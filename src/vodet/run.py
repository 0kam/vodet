from python.gmvae import GMVAE
from torchvision import transforms


data_dirs = {
    "train":"/home/okamoto/moni1000/data/d4_small/train",
    "validation":"/home/okamoto/moni1000/data/d4_small/validation",
    "unlabelled":"/home/okamoto/moni1000/data/d4_small/unlabelled"
    }

gmvae = GMVAE(data_dirs)

gmvae.set_patches("VoTT")

transform = \
    {"labelled":transforms.Compose(
        [transforms.Resize((24,24)),
        transforms.RandomHorizontalFlip(),
        transforms.ColorJitter(brightness=0.5,contrast=0.3),
        transforms.ToTensor()
        ]),
    "unlabelled":transforms.Compose(
        [transforms.Resize((24,24)),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor()
        ]),
    "validation":transforms.Compose(
        [transforms.Resize((24,24)),
        transforms.ToTensor()
        ])
    }

gmvae.set_dataloaders(batch_size=512, transform=transform)
gmvae.set_model(z_dim=128, device="cuda:0")
gmvae.train(50)

d = gmvae.detector(label_type="VoTT", step_ratio = 0.25, iou_th=0.05)
import os 
os.mkdir("data/d4_small/unlabelled/detected")
in_path = "data/d4_small/unlabelled/source/IMAG0382.JPG"

num = d.detect_img(in_path=in_path, out_path=in_path.replace("source", "detected"))




data_dirs = {
    "train":"/home/okamoto/moni1000/data/dai4sekkei/train",
    "validation":"/home/okamoto/moni1000/data/dai4sekkei/validation",
    "unlabelled":"/home/okamoto/moni1000/data/dai4sekkei/unlabelled"
    }

gmvae = GMVAE(data_dirs)

transform = \
    {"labelled":transforms.Compose(
        [transforms.Resize((24,24)),
        transforms.RandomHorizontalFlip(),
        #transforms.ColorJitter(brightness=0.5,contrast=0.3),
        transforms.ToTensor()
        ]),
    "unlabelled":transforms.Compose(
        [transforms.Resize((24,24)),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor()
        ]),
    "validation":transforms.Compose(
        [transforms.Resize((24,24)),
        transforms.ToTensor()
        ])
    }

gmvae.set_dataloaders(batch_size=512, transform=transform)
gmvae.set_model(z_dim=512, device="cuda:0")
gmvae.train(30, precision_th=95.0)

d = gmvae.detector(label_type="labelme", conf_th=0.99, step_ratio = 0.5, iou_th=0.05)

import os 
os.mkdir("data/dai4sekkei/unlabelled/detected")
in_path = "data/dai4sekkei/test/source/IMAG0637.JPG"

num = d.detect_img(in_path=in_path, out_path=in_path.replace("source", "detected"))


from python.utils import exif_date
dir_2017 = "data/dai4sekkei/croped/2017"
dir_2018 = "data/dai4sekkei/croped/2018"

df_2017 = d.detect_dir(dir_2017, dir_2017.replace("croped", "detected"))
df_2017.to_csv("data/dai4sekkei/detected/df_2017.csv")
date = exif_date("data/dai4sekkei/source/2017")
d.draw_barplot(date, "data/dai4sekkei/detected/2017.png")

import pandas as pd

df_2018 = d.detect_dir(dir_2018, dir_2018.replace("croped", "detected"))
df_2018.to_csv("data/dai4sekkei/detected/df_2018_2.csv")

date = exif_date("data/dai4sekkei/source/2018")
df = d.draw_barplot(date, "data/dai4sekkei/detected/2018.png")
df.to_csv("data/dai4sekkei/detected/df_2018.csv")
