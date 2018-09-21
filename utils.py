import numpy as np
import cv2
import torch, torchvision
import torch.nn as nn
import torch.nn.functional as F
# from PIL import Image
from torch.nn.modules.utils import _pair

def generate_random_mask(width, height, lines=1, spots=1, ellipses=1, scale=5.0):
    mask = np.zeros((height, width, 1), dtype=np.uint8)
    scale = max(width, height)*scale/100

    # draw lines
    ## start
    if lines > 0:
        x0 = np.random.randint(0, width, size=lines)
        y0 = np.random.randint(0, height, size=lines)
        ## end
        x1 = np.random.randint(0, width, size=lines)
        y1 = np.random.randint(0, height, size=lines)

        thickness = np.random.rand(lines)*scale
        thickness = (thickness.clip(1, int(scale)+1)).astype(int)
        for idx in range(lines):
            cv2.line(mask, 
                     (x0[idx], y0[idx]), 
                     (x1[idx], y1[idx]), 
                     1, 
                     thickness[idx]
                    )

    # draw spots
    if spots > 1:
        x0 = np.random.randint(0, width, size=spots)
        y0 = np.random.randint(0, height, size=spots)
        radius = (np.random.rand(spots)*scale*2).clip(1, 2*int(scale)+1).astype(int)
        
        for idx in range(spots):
            cv2.circle(mask, (x0[idx], y0[idx]), radius[idx], 1, -1)
        
        
    # draw ellipse
    if ellipses > 1:
        x = np.random.randint(width//2-width//4, width//2+width//4, size=ellipses)
        y = np.random.randint(height//2-height//4, height//2+height//4, size=ellipses)
        
        a = np.random.randint(1, width, size=ellipses)
        b = np.random.randint(1, height, size=ellipses)
        
        t1 = np.random.randint(0, 180, size=ellipses)
        t2 = np.random.randint(0, 90, size=ellipses)
        t3 = np.random.randint(1, 90, size=ellipses)
        
        thickness = np.random.rand(ellipses)*scale
        thickness = (thickness.clip(1, int(scale)+1)).astype(int)
        for idx in range(ellipses):
            cv2.ellipse(mask, (x[idx], y[idx]), (a[idx], b[idx]), 
                       t1[idx], t2[idx], t2[idx]+t3[idx], 1, thickness[idx])
    
    mask = mask[:,:,0].T
    
    return 1-mask
    
def conv_down_size(size, kernel_size, stride, padding):
    out = (size - kernel_size +2*padding)/stride + 1
    
    if isinstance(out, int) or out == int(out):
        return int(out)
    else:
        raise RuntimeError("New size is not an integer.")

def deconv_up_size(size, kernel_size, stride, padding, outpad=0):
    return (size-1)*stride + kernel_size - 2*padding + outpad
    
    

class PartialConv2d(nn.modules.conv._ConvNd):
    def __init__(self, in_channels, out_channels, kernel_size, 
                 stride=1, padding=0, dilation=1, groups=1, bias=True):
        
        with torch.no_grad():
            self.mask_weight = torch.ones(1, # out channel
                                          1, # in channel
                                          kernel_size, 
                                          kernel_size,
                                          dtype=torch.float
                                         )/(kernel_size)**2
            self.mask_weight = self.mask_weight.to(device)
            
#             print(self.mask_weight.dtype)
        
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