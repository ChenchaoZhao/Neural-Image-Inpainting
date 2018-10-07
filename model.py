import numpy as np
import cv2
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.modules.utils import _pair
import torchvision
from torchvision import models



device = torch.device("cuda" if torch.cuda.is_available() else "cpu")    

class PartialConv2d(nn.modules.conv._ConvNd):
    def __init__(self, in_channels, out_channels, kernel_size, 
                 stride=1, padding=0, dilation=1, groups=1, bias=True,
                 device=device):
        
        with torch.no_grad():
            self.mask_weight = torch.ones(1, # out channel
                                          1, # in channel
                                          kernel_size, 
                                          kernel_size,
                                          dtype=torch.float
                                         )/(kernel_size)**2
            self.mask_weight = self.mask_weight.to(device)

        
        kernel_size = _pair(kernel_size)
        stride = _pair(stride)
        padding = _pair(padding)
        dilation = _pair(dilation)
        super(PartialConv2d, self).__init__(
            in_channels, out_channels, kernel_size, stride, padding, dilation,
            False, _pair(0), groups, bias)

    def forward(self, image, mask):
        assert mask.shape[0] == 1
        assert mask.shape[1] == 1
        assert image.shape[1] >= image.shape[1]
        assert image.shape[2:] == mask.shape[2:] # image: (batch, channel, w, h)
        with torch.no_grad():
            mask_conv = F.conv2d(mask.float(), self.mask_weight, None, self.stride,
                            self.padding, self.dilation, self.groups)
            new_mask = mask_conv > 0
        
        image_hole = image * mask.float() # 0 for hole pixels, 1 for non-hole
        image_conv = F.conv2d(image_hole, self.weight, self.bias, self.stride,
                        self.padding, self.dilation, self.groups)
        
        image_conv = image_conv * mask_conv # re-weight the non-hole pixels
        
        
        return image_conv, new_mask        

class PConvBlock(nn.Module):
    def __init__(self, in_channel, out_channel, conv_para, pool_para):
        super(PConvBlock, self).__init__()
        self.pconv = PartialConv2d(in_channel, out_channel, **conv_para)
        self.bn = nn.BatchNorm1d(out_channel)
        self.relu = nn.ReLU()
        self.pool = nn.MaxPool2d(**pool_para)
    def forward(self, img, msk):
        img, msk = self.pconv(img, msk) 
        msk_vec = msk.view(1,1,-1)
        batch_size, channel, _, _ = img.size()
        img_vec = img.view(batch_size, channel, -1)
        img_vec = img_vec[:,:, msk_vec[0,0,:] > 0] # take non-hole pixels
        img_vec = self.bn(img_vec)
        img[:,:,msk[0,0,:,:]>0] = img_vec
        img = self.relu(img)
        img = self.pool(img)
        msk = self.pool(msk.float())
        return img, msk

class PConvNet(nn.Module):
    def __init__(self, n_hidden=8):
        super(PConvNet, self).__init__()
        para310 = {"kernel_size": 3, "stride": 1}
        para220 = {"kernel_size": 2, "stride": 2}
        para210 = {"kernel_size": 2, "stride": 1}
        
        self.down_ = []
        self.pconv0 = self._get_pconv_block(3, n_hidden, para310, para220)
        self.down_.append(self.pconv0)
        self.pconv1 = self._get_pconv_block(n_hidden, n_hidden, para310, para210)
        self.down_.append(self.pconv1)
        self.pconv2 = self._get_pconv_block(n_hidden, 2*n_hidden, para310, para220)
        self.down_.append(self.pconv2)
        self.pconv3 = self._get_pconv_block(2*n_hidden, 2*n_hidden, para310, para210)
        self.down_.append(self.pconv3)
        self.pconv4 = self._get_pconv_block(2*n_hidden, 4*n_hidden, para310, para220)
        self.down_.append(self.pconv4)
        self.pconv5 = self._get_pconv_block(4*n_hidden, 8*n_hidden, para310, para220)
        self.down_.append(self.pconv5)
        self.pconv6 = self._get_pconv_block(8*n_hidden, 8*n_hidden, para310, para210)
        self.down_.append(self.pconv6)
        self.pconv7 = self._get_pconv_block(8*n_hidden, 16*n_hidden, para310, para220)
        self.down_.append(self.pconv7)
        self.up_ = []
        self.tconv7, self.comp7 = self._get_tconv_block(8*n_hidden, 16*n_hidden, para310, para220)
        self.up_.append((self.tconv7, self.comp7))
        self.tconv6, self.comp6 = self._get_tconv_block(8*n_hidden, 8*n_hidden, para310, para210)
        self.up_.append((self.tconv6, self.comp6))
        self.tconv5, self.comp5 = self._get_tconv_block(4*n_hidden, 8*n_hidden, para310, para220)
        self.up_.append((self.tconv5, self.comp5))
        self.tconv4, self.comp4 = self._get_tconv_block(2*n_hidden, 4*n_hidden, para310, para220)
        self.up_.append((self.tconv4, self.comp4))
        self.tconv3, self.comp3 = self._get_tconv_block(2*n_hidden, 2*n_hidden, para310, para210)
        self.up_.append((self.tconv3, self.comp3))
        self.tconv2, self.comp2 = self._get_tconv_block(n_hidden, 2*n_hidden, para310, para220)
        self.up_.append((self.tconv2, self.comp2))
        self.tconv1, self.comp1 = self._get_tconv_block(n_hidden, n_hidden, para310, para210)
        self.up_.append((self.tconv1, self.comp1))
        self.tconv0, self.comp0 = self._get_tconv_block(3, n_hidden, para310, para220, activation="sigmoid")
        self.up_.append((self.tconv0, self.comp0))
    def encoder(self, img, msk):
        img_ = [img]
        msk_ = [msk]
        for pconv in self.down_:
            img, msk = pconv(img, msk)          
            img_.append(img)
            msk_.append(msk.float())
        
        return img_, msk_
    
    def decoder(self, img_, msk_):
        feature_maps_ = []
        idx = -1
        img = img_[idx]
        for (tconv, comp) in self.up_:
            idx -= 1
            img = tconv(img)
            img = torch.cat((img, img_[idx]), dim=1)
            img = comp(img)
        return img
    
    def _get_pconv_block(self, in_channel, out_channel, conv_para, pool_para):
        return PConvBlock(in_channel, out_channel, conv_para, pool_para)
    def _get_tconv_block(self, out_channel, in_channel, tconv_para, upsam_para, 
                         activation="relu"):
        upsample = nn.ConvTranspose2d(in_channel, in_channel, **upsam_para)
        bn = nn.BatchNorm2d(in_channel)
        tconv = nn.ConvTranspose2d(in_channel, out_channel, **tconv_para)
        if activation=="relu":
            act = nn.ReLU()
        elif activation == "sigmoid":
            act = nn.Sigmoid()
        else:
        	raise ValueError("activation function should be relu or sigmoid")
        # then channel compression
        comp = nn.Conv2d(2*out_channel, out_channel, 1)
        seq_ = nn.Sequential(upsample, bn, tconv, act)
        return seq_, comp
    
    def forward(self, img, msk):
        img = img*msk.float()
        img_, msk_ = self.encoder(img, msk)
        out = self.decoder(img_, msk_)
        return out


class SpacenetDataset(torch.utils.data.Dataset):
    def __init__(self, train=True):
        if train:
            path = "data/test/"
        else:
            path = "data/train/"
            
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
        img = Image.open(self.path+self.filelist[idx])
        return self.transform(img)

    def __len__(self):
        return len(self.filelist)
    
    def __repr__(self):
        return "Paris {} Images".format(self.__len__())

class FeatureMaps(nn.Module):
    def __init__(self, select=['4', '9', '16']):
        """Select pool1 pool2 pool3."""
        super(FeatureMaps, self).__init__()
        self.select = select
        self.vgg = models.vgg16(pretrained=True).features
        
    def forward(self, x):
        """Extract multiple pool feature maps."""
        features = []
        for name, layer in self.vgg._modules.items():
            x = layer(x)
            if name in self.select:
                features.append(x)
        return features

def gram_matrix(feat):
    batch, channel, _, _ = feat.shape
    vector = feat.reshape((batch, channel, -1))
    gram_mat = torch.zeros((batch, channel, channel))
    for bdx in range(batch):
        v = vector[bdx, :, :]
        gram_mat[bdx, :, :] = torch.mm(v, v.t())
    return gram_mat

class LpLoss(nn.modules.loss._Loss):
    """
    take Lp norm of predict - target
    
    """
    def __init__(self, p, reduction='elementwise_mean'):
        
        super(LpLoss, self).__init__(reduction=reduction)
        assert p >= 0
        self.p = p
        self.reduction = reduction

    def forward(self, predict, target):
        assert not target.requires_grad
        shape_of_input = predict.shape
        predict = predict.reshape(predict.shape[0], -1)
        target = target.reshape(target.shape[0], -1)
        loss = predict - target
        loss = loss.norm(dim=1, p=self.p)**self.p
        if self.reduction == 'elementwise_mean':
            loss = loss.mean(dim=0)
        elif self.reduction == 'sum':
            loss = loss.sum(dim=0)
        return loss