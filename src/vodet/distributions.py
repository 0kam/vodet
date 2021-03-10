from torch import nn, optim
import torch
from torch.nn import functional as F
from tensorboardX import SummaryWriter

import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

from pixyz.distributions import Normal, RelaxedCategorical
from pixyz.models import Model
from pixyz.losses import ELBO

# distributions for labeled data
## inference model q(z|x,y)

# distributions for labeled data
## inference model q(z|x,y)
class Inference(Normal):
    def __init__(self, z_dim, y_dim):
        super().__init__(cond_var=["x","y"], var=["z"], name="q")

        self.conv2d_1 = nn.Conv2d(3, 16, 2)
        self.prelu1 = nn.PReLU()
        self.conv2d_2 = nn.Conv2d(16, 32, 2)
        self.prelu2 = nn.PReLU()
        self.fc1 = nn.Linear(1152, 512)
        self.prelu3 = nn.PReLU()
        self.fc2 = nn.Linear(512 + y_dim, 512)
        self.prelu4 = nn.PReLU()
        self.fc31 = nn.Linear(512, z_dim)
        self.fc32 = nn.Linear(512, z_dim)

    def forward(self, x, y):
        bs = x.size()[0]
        h = self.prelu1(self.conv2d_1(x))
        h = F.max_pool2d(h, 2, padding = 1)
        h = self.prelu2(self.conv2d_2(h))
        h = F.max_pool2d(h, 2, padding = 1)
        h = h.view(bs, -1)
        h = self.prelu4(self.fc1(h))
        h = F.relu(self.fc2(torch.cat([h, y], 1)))
        return {"loc": self.fc31(h), "scale": F.softplus(self.fc32(h))}

# generative model p(x|z) 
class Generator(Normal):
    def __init__(self, z_dim):
        super().__init__(cond_var=["z"], var=["x"], name="p")

        self.fc1 = nn.Linear(z_dim, 256)
        self.prelu1 = nn.PReLU()
        self.fc2 = nn.Linear(256, 512)
        self.prelu2 = nn.PReLU()
        self.fc3 = nn.Linear(512, 1152)
        self.prelu3 = nn.PReLU()
        self.convt2d_1 = nn.ConvTranspose2d(32, 16, 2, 2)
        self.prelu4 = nn.PReLU()
        self.convt2d_21 = nn.ConvTranspose2d(16, 3, 2, 2)
        self.convt2d_22 = nn.ConvTranspose2d(16, 3, 2, 2)

    def forward(self, z):
        bs = z.size()[0]
        h = self.prelu1(self.fc1(z))
        h = self.prelu2(self.fc2(h))
        h = self.prelu3(self.fc3(h))
        h = h.view(bs, 32, 6, 6)
        h = self.prelu4(self.convt2d_1(h))
        return {"loc": torch.sigmoid(self.convt2d_21(h)), "scale":F.softplus(self.convt2d_22(h))}

# classifier p(y|x)
class Classifier(RelaxedCategorical):
    def __init__(self, y_dim):
        super(Classifier, self).__init__(cond_var=["x"], var=["y"], name="p")
        self.conv2d_1 = nn.Conv2d(3, 16, 2)
        self.prelu1 = nn.PReLU()
        self.conv2d_2 = nn.Conv2d(16, 32, 2)
        self.prelu2 = nn.PReLU()
        self.fc1 = nn.Linear(1152, 512)
        self.prelu3 = nn.PReLU()
        self.fc2 = nn.Linear(512, y_dim)

    def forward(self, x):
        bs = x.size()[0]
        h = self.conv2d_1(x)
        h = self.prelu1(h)
        h = F.max_pool2d(h, 2, padding = 1)
        h = self.conv2d_2(h)
        h = self.prelu2(h)
        h = F.max_pool2d(h, 2, padding = 1)
        h = h.view(bs, -1)
        h = self.fc1(h)
        h = self.prelu3(h)
        h = F.softmax(self.fc2(h), dim=1)
        h = h + 1e-7
        return {"probs": h}


# prior model p(z|y)
class Prior(Normal):
    def __init__(self, z_dim, y_dim):
        super().__init__(var=["z"], cond_var=["y"], name="p_{prior}")

        self.fc1 = nn.Linear(y_dim, 32)
        self.prelu1 = nn.PReLU()
        self.fc21 = nn.Linear(32, z_dim)
        self.fc22 = nn.Linear(32, z_dim)
    
    def forward(self, y):
        h = self.prelu1(self.fc1(y))
        return {"loc": self.fc21(h), "scale": F.softplus(self.fc22(h))}

