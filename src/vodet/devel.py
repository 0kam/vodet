from PIL import Image
from glob import glob
import os
import shutil

if os.path.exists("data/tomakomai_small/croped"):
    shutil.rmtree("data/tomakomai_small/croped")   
os.mkdir("data/tomakomai_small/croped")

files = glob("data/tomakomai_small/source/*")

for f in files:
    im = Image.open(f)
    out_f = f.replace("source", "croped")
    im.crop((310, 1450, 570, 1580)).save(out_f)


from vodet.gmvae import GMVAE
from vodet.utils import exif_date
from torchvision import transforms
import pandas as pd

data_dirs = {
    "train":"/home/okamoto/moni1000/data/tomakomai_small/train",
    "validation":"/home/okamoto/moni1000/data/tomakomai_small/validation",
    "unlabelled":"/home/okamoto/moni1000/data/tomakomai_small/unlabelled"
    }

gmvae = GMVAE(data_dirs)

gmvae.set_patches("labelme")

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

gmvae.set_dataloaders(batch_size=128, transforms=transform)
gmvae.set_model(z_dim=8, device="cuda:0")
gmvae.train(epochs=50, precision_th=95.0)

image_path = "/home/okamoto/moni1000/data/tomakomai_small/croped/2013080912.jpg"
labels_path = "/home/okamoto/moni1000/data/tomakomai_small/train/labels"