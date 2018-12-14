import os
import numpy as np
import cv2
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.modules.utils import _pair
from torchvision import transforms
from torchvision import models



class PartialConv2d(nn.modules.conv._ConvNd):
    """Perform Partial Convolution over the input image with a given mask

    Parameters:

    - in_channels (int) – Number of channels in the input image
    - out_channels (int) – Number of channels produced by the convolution
    - kernel_size (int or tuple) – Size of the convolving kernel
    - stride (int or tuple, optional) – Stride of the convolution. Default: 1
    - padding (int or tuple, optional) – Zero-padding added to both sides of 
      the input. Default: 0
    - dilation (int or tuple, optional) – Spacing between kernel elements. 
      Default: 1
    - groups (int, optional) – Number of blocked connections from input channels 
      to output channels. Default: 1
    - bias (bool, optional) – If True, adds a learnable bias to the output. 
      Default: True
    - device (class torch.device, optional) The device on which the mask tensor
      will be allocated. 
      Default: device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


    Shape:

    - Input: image (batch, in_channel, height, width) and mask (1, 1, height, width)
    - Output: feature map (batch, out_channel, new_height, new_width) and mask 
      (1, 1, new_height, new_width)

    Variables:

    - weight (Tensor) – the learnable weights of the module of shape (out_channels, 
      in_channels, kernel_size[0], kernel_size[1])
    - bias (Tensor) – the learnable bias of the module of shape (out_channels)

    """

    def __init__(self, in_channels, out_channels, kernel_size, stride=1,
                 padding=0, dilation=1, groups=1, bias=True, eps=0):

        kernel_size = _pair(kernel_size)
        stride = _pair(stride)
        padding = _pair(padding)
        dilation = _pair(dilation)
        super(PartialConv2d, self).__init__(
            in_channels, out_channels, kernel_size, stride, padding, dilation,
            False, _pair(0), groups, bias)
        self.pool = nn.AvgPool2d(kernel_size, stride, padding)
        self.eps = eps

    def forward(self, image, mask):
        assert mask.shape[0] == 1
        assert mask.shape[1] == 1
        assert image.shape[2:] == mask.shape[2:]  # image: (batch, channel, w, h)
        with torch.no_grad():
            mask_conv = self.pool(mask.float())
            new_mask = mask_conv > self.eps

        image_hole = image * mask.float()  # 0 for hole pixels, 1 for non-hole
        image_conv = F.conv2d(image_hole, self.weight, self.bias, self.stride,
                              self.padding, self.dilation, self.groups)

        image_conv = image_conv * mask_conv  # re-weight the non-hole pixels

        return image_conv, new_mask


class PConvBlock(nn.Module):
    """The input image and mask are passed to partial convolution and then 
    partial batch normalization (avoiding the hole region).  
    The output feature map is then passed to a ReLU layer and MaxPool2d layer; 
    the mask is downsampled with the same MaxPool2d layer.

    Parameters:

    - in_channels (int) – Number of channels in the input image
    - out_channels (int) – Number of channels produced by the convolution
    - conv_para (dict) – Parameters of partial convolution layer
    - pool_para (dict) – Paramters of max-pooling layer, see 
      class torch.nn.MaxPool2d
    Shape:
    - Input: image (batch, in_channel, height, width) and mask 
      (1, 1, height, width)g
    - Output: feature map (batch, out_channel, new_height, new_width) 
      and mask (1, 1, new_height, new_width)
    Variables:
    - weight (Tensor) – the learnable weights of the module of shape 
      (out_channels, in_channels, kernel_size[0], kernel_size[1])
    - bias (Tensor) – the learnable bias of the module of shape (out_channels)

    """

    def __init__(self, in_channel, out_channel, conv_para, pool_para):
        super(PConvBlock, self).__init__()
        self.pconv = PartialConv2d(in_channel, out_channel, **conv_para)
        self.bn = nn.BatchNorm1d(out_channel)
        self.relu = nn.ReLU()
        self.pool = nn.MaxPool2d(**pool_para)

    def forward(self, img, msk):
        img, msk = self.pconv(img, msk)
        msk_vec = msk.view(1, 1, -1)
        batch_size, channel, _, _ = img.size()
        img_vec = img.view(batch_size, channel, -1)
        img_vec = img_vec[:, :, msk_vec[0, 0, :] > 0]  # take non-hole pixels
        img_vec = self.bn(img_vec)
        img[:, :, msk[0, 0, :, :] > 0] = img_vec
        img = self.relu(img)
        img = self.pool(img)
        msk = self.pool(msk.float())
        return img, msk

class TConvBlock(nn.Module):
    """2D Tranposed Convolution Block
    Upsample (TConv2d)
    Batch Norm
    ReLU
    Dropout
    TConv1d
    Compression (1x1 Conv2d) if not last block
    """
    def __init__(self, out_channel, in_channel, tconv_para, upsam_para, 
                 is_last=False, dropout=0):
        super(TConvBlock, self).__init__()
        self.upsample = nn.ConvTranspose2d(in_channel, in_channel, **upsam_para)
        self.bn = nn.BatchNorm2d(in_channel)
        self.relu = nn.LeakyReLU()
        self.dropout = nn.Dropout(dropout)
        self.tconv = nn.ConvTranspose2d(in_channel, out_channel, **tconv_para)
        self.is_last = is_last
        if not is_last:
            self.compress = nn.Conv2d(2*out_channel, out_channel, kernel_size=1)
        
    def forward(self, img, mirror=None):
        """forward pass
        input: seq, mirror
        seq.size() == mirror.size()
        input shape: (batch, channel, seq), i.e. Conv1D convention
        """
        img = self.upsample(img) # undo average pooling
        img = self.bn(img)
        img = self.relu(img) # nonlinearity
        img = self.dropout(img)
        img = self.tconv(img) # transposed conv
        
        if mirror is None:
            if self.is_last:
                return img
            else:
                raise ValueError("Please input mirror seq")
        img = torch.cat((img, mirror), dim=1) # concat seq and mirror in channel dim
        img = self.compress(img) # compress: 2*out_channel -> out_channel
        return img     

class PConvNet(nn.Module):
    def __init__(self, n_hidden):
        super(PConvNet, self).__init__()
        para310 = {"kernel_size": 3, "stride": 1}
        para220 = {"kernel_size": 2, "stride": 2}
        para210 = {"kernel_size": 2, "stride": 1}
        
        self.down_ = []
        self.pconv0 = PConvBlock(3, n_hidden, para310, para220)
        self.down_.append(self.pconv0)
        self.pconv1 = PConvBlock(n_hidden, n_hidden, para310, para210)
        self.down_.append(self.pconv1)
        self.pconv2 = PConvBlock(n_hidden, 2*n_hidden, para310, para220)
        self.down_.append(self.pconv2)
        self.pconv3 = PConvBlock(2*n_hidden, 2*n_hidden, para310, para210)
        self.down_.append(self.pconv3)
        self.pconv4 = PConvBlock(2*n_hidden, 4*n_hidden, para310, para220)
        self.down_.append(self.pconv4)
        self.pconv5 = PConvBlock(4*n_hidden, 8*n_hidden, para310, para220)
        self.down_.append(self.pconv5)
        self.pconv6 = PConvBlock(8*n_hidden, 8*n_hidden, para310, para210)
        self.down_.append(self.pconv6)
        self.pconv7 = PConvBlock(8*n_hidden, 16*n_hidden, para310, para220)
        self.down_.append(self.pconv7)
        self.up_ = []
        self.tconv7 = TConvBlock(8*n_hidden, 16*n_hidden, para310, para220)
        self.up_.append(self.tconv7)
        self.tconv6 = TConvBlock(8*n_hidden, 8*n_hidden, para310, para210)
        self.up_.append(self.tconv6)
        self.tconv5 = TConvBlock(4*n_hidden, 8*n_hidden, para310, para220)
        self.up_.append(self.tconv5)
        self.tconv4 = TConvBlock(2*n_hidden, 4*n_hidden, para310, para220)
        self.up_.append(self.tconv4)
        self.tconv3 = TConvBlock(2*n_hidden, 2*n_hidden, para310, para210)
        self.up_.append(self.tconv3)
        self.tconv2 = TConvBlock(n_hidden, 2*n_hidden, para310, para220)
        self.up_.append(self.tconv2)
        self.tconv1 = TConvBlock(n_hidden, n_hidden, para310, para210)
        self.up_.append(self.tconv1)
        self.tconv0 = TConvBlock(3, n_hidden, para310, para220, is_last=True)
        self.up_.append(self.tconv0)
    
    def encoder(self, img, msk):
        img_ = [img]
        msk_ = [msk]
        for pconv in self.down_:
            img, msk = pconv(img, msk)          
            img_.append(img)
            msk_.append(msk.float())
        return img_, msk_
    
    def decoder(self, img_, msk_):
        
        img_ = img_[::-1]
        msk_ = msk_[::-1]
        
        feature_maps_ = []
        img = img_[0]
        for idx, tconv in enumerate(self.up_):
            if idx+1 < len(self.up_):
                img = tconv(img, img_[idx+1])
            else:
                img = tconv(img)

        return img
    
    def forward(self, img, msk):
        batch, channel, width, height = img.size()
        img = img.view(batch, channel, -1)
        msk = msk.view(1,1,-1)
        self._mean = img[:,:,msk[0,0,:]>0].mean(dim=2, keepdim=True)
        self._std = 3*(img[:,:,msk[0,0,:]>0].std(dim=2, keepdim=True)+1e-6)
        img[:,:,msk[0,0,:]>0] = (img[:,:,msk[0,0,:]>0] - self._mean)/(self._std)
        img = img.view(batch, channel, width, height)
        msk = msk.view(1,1,width, height)
        
        img_, msk_ = self.encoder(img, msk)
        out = self.decoder(img_, msk_)
        
        out = out.view(batch, channel, -1)
        out = out*self._std + self._mean
        out = out.view(batch, channel, width, height)
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
        img = Image.open(self.path + self.filelist[idx])
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
