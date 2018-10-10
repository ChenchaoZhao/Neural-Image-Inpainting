from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import cv2

import torch
import torch.nn as nn
import torch.nn.functional as F

import torchvision
import torchvision.transforms as transforms
from torchvision import models

from utils import *
from model import *

device = torch.device('cuda' if torch.cuda.is_available() else "cpu")

model = PConvNet(n_hidden=8).to(device).eval()
weight_dict = torch.load("model_weight.ckpt")
model.load_state_dict(weight_dict)

data_test = SpacenetDataset(train=False)

in_img = data_test[0].to(device)

mask = generate_random_mask(256, 256, 0, 5, 0, 3)
mask = torch.from_numpy(mask).reshape(1,1,256,256)
mask = mask.float()
mask = mask.to(device)

out_img = model(in_img.unsqueeze(0), mask)
out_img = polish_output(in_img, out_img[0,:,:,:], mask, 12)
out_img = polish_output(in_img, out_img, mask, 12)

# output
imshow(pol_img, brighten=1.5)
# masked image
imshow(in_img, mask, brighten=1.5)
# ground truth
imshow(in_img, brighten=1.5)

