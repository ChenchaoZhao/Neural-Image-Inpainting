from PIL import Image
import numpy as np
import cv2
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from torchvision import models
import torchvision.transforms as transforms
from utils import *
from model import *

device = torch.device('cuda' if torch.cuda.is_available() else "cpu")

n_epoch = 500
w_hole = 7
w_feature_map = 1e-2
w_style = 0.05
mask_scale = 15
batch_size = 10

class SpacenetDataset(torch.utils.data.Dataset):
    def __init__(self, train=True):
        if train:
            path = "data/train/"
        else:
            path = "data/test/"

        self.path = path
        self.filelist = os.listdir(path)
        mean_ = [0.2225, 0.3023, 0.2781]
        std_ = [0.1449, 0.1839, 0.1743]
        self.transform = transforms.Compose([
            transforms.Resize((256, 256)),
            transforms.ToTensor(),
            transforms.Normalize(mean_, std_)
        ])

    def __getitem__(self, idx):
        img = Image.open(self.path + self.filelist[idx])
        return self.transform(img)

    def __len__(self):
        return len(self.filelist)

    def __repr__(self):
        return "Paris {} Images".format(self.__len__())

# load train data
data_train = SpacenetDataset(train=True)

model = PConvNet(n_hidden=8).to(device)

c1_hole = nn.L1Loss()
c1_nonh = nn.L1Loss()
c2_hole = nn.MSELoss()
c2_nonh = nn.MSELoss()

feature_map = FeatureMaps(['4', '9', '16']).to(device).eval()

c1_feat_map_0 = nn.L1Loss()
c1_feat_map_1 = nn.L1Loss()
c1_feat_map_2 = nn.L1Loss()

c1_comp_map_0 = nn.L1Loss()
c1_comp_map_1 = nn.L1Loss()
c1_comp_map_2 = nn.L1Loss()

c1_style_loss_0 = nn.L1Loss()
c1_style_loss_1 = nn.L1Loss()
c1_style_loss_2 = nn.L1Loss()

c1_comp_loss_0 = nn.L1Loss()
c1_comp_loss_1 = nn.L1Loss()
c1_comp_loss_2 = nn.L1Loss()

optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
loss_mean = []
loss_std = []

train_loader = torch.utils.data.DataLoader(data_train, 
                                           batch_size=batch_size, 
                                           shuffle=True)

model.train()

for edx in range(n_epoch):
    loss_ = []
    hole_loss_ = []
    nonh_loss_ = []
    feat_loss_ = []
    styl_loss_ = []
    for image in train_loader:
        mask = generate_random_mask(256, 256, 1, 5, 5, mask_scale)
        hole_fraction = 1 - mask.mean()
        mask = torch.from_numpy(mask).reshape(1,1,256,256)
        mask = mask.float()
        mask = mask.to(device)
        
        image = image.to(device)
        output = model(image, mask)
        
        feature_y = feature_map(image)
        feature_x = feature_map(output)
        feature_c = feature_map(image*mask + output*(1-mask))
        
        
        fm_loss  = c1_feat_map_0(feature_x[0], feature_y[0].detach())/(64*128**2)
        gram_out = gram_matrix(feature_x[0]).to(device)
        gram_in = gram_matrix(feature_y[0].detach()).to(device)
        style_loss = c1_style_loss_0(gram_out, gram_in)/64/128**2
        
        fm_loss  += c1_comp_map_0(feature_c[0], feature_y[0].detach())/(64*128**2)
        gram_out = gram_matrix(feature_c[0]).to(device)
        gram_in = gram_matrix(feature_y[0].detach()).to(device)
        style_loss += c1_comp_loss_0(gram_out, gram_in)/64/128**2
        
        fm_loss += c1_feat_map_1(feature_x[1], feature_y[1].detach())/(128*64**2)
        gram_out = gram_matrix(feature_x[1]).to(device)
        gram_in = gram_matrix(feature_y[1].detach()).to(device)
        style_loss += c1_style_loss_1(gram_out, gram_in) /128/64**2
        
        fm_loss += c1_comp_map_1(feature_c[1], feature_y[1].detach())/(128*64**2)
        gram_out = gram_matrix(feature_c[1]).to(device)
        gram_in = gram_matrix(feature_y[1].detach()).to(device)
        style_loss += c1_comp_loss_1(gram_out, gram_in) /128/64**2
        
        fm_loss += c1_feat_map_2(feature_x[2], feature_y[2].detach())/(256*32**2)
        gram_out = gram_matrix(feature_x[2]).to(device)
        gram_in = gram_matrix(feature_y[2].detach()).to(device)
        style_loss += c1_style_loss_1(gram_out, gram_in) /256/32**2
        
        fm_loss += c1_comp_map_2(feature_c[2], feature_y[2].detach())/(256*32**2)
        gram_out = gram_matrix(feature_c[2]).to(device)
        gram_in = gram_matrix(feature_y[2].detach()).to(device)
        style_loss += c1_comp_loss_2(gram_out, gram_in) /256/32**2
        
        # pixel-wise loss
        l1_hole = c1_hole(output*(1-mask), image*(1-mask))
        l1_nonh = c1_nonh(output*mask, image*mask)
        l2_hole = c2_hole(output*(1-mask), image*(1-mask))
        l2_nonh = c2_nonh(output*mask, image*mask)
        
        hole_loss = (l1_hole+l2_hole) * w_hole/256**2 / (hole_fraction + 1e-5)
        nonh_loss = (l1_nonh+l2_nonh)/256**2
        feat_loss = w_feature_map * fm_loss
        styl_loss = w_style * style_loss
        loss = hole_loss + nonh_loss + feat_loss + styl_loss
        
        loss_.append(loss.item())
        hole_loss_.append(hole_loss.item())
        nonh_loss_.append(nonh_loss.item())
        feat_loss_.append(feat_loss.item())
        styl_loss_.append(styl_loss.item())

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    
    print("{}/{}: loss = {:3f} ± {:3f}".format(edx+1, n_epoch, np.mean(loss_), 
                                       np.std(loss_)))
    print("{}/{}: hole = {:3f} ± {:3f}".format(edx+1, n_epoch, np.mean(hole_loss_)/np.mean(loss_), 
                                       np.std(hole_loss_)/np.std(loss_)))
    print("{}/{}: nonh = {:3f} ± {:3f}".format(edx+1, n_epoch, np.mean(nonh_loss_)/np.mean(loss_), 
                                       np.std(nonh_loss_)/np.std(loss_)))
    print("{}/{}: feat = {:3f} ± {:3f}".format(edx+1, n_epoch, np.mean(feat_loss_)/np.mean(loss_), 
                                       np.std(feat_loss_)/np.std(loss_)))
    print("{}/{}: styl = {:3f} ± {:3f}".format(edx+1, n_epoch, np.mean(styl_loss_)/np.mean(loss_), 
                                       np.std(styl_loss_)/np.std(loss_)))
  
    loss_mean.append(np.mean(loss_))
    loss_std.append(np.std(loss_))
    torch.save(model.state_dict(), "model_weight.ckpt")
    
    if (edx+1)%10 == 0:
        torch.save(model.state_dict(), "model_weight_ep{}.ckpt".format(edx+1))
        
