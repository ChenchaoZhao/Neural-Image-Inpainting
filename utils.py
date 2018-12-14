import numpy as np
import cv2
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision



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
    

def paint_mask(list_of_coords_radii, size=(256, 256)):
    mask = np.zeros(size)
    for x, y, r in list_of_coords_radii:
        mask = cv2.circle(mask, (x, y), r, 1, -1)
    return 1 - mask.clip(0,1)
def imshow(torch_image, mask=None, brighten = 1):
    
    img = torch_image.detach().cpu().permute((1,2,0))
    if mask is not None:
        mask = mask.cpu()
    img -= img.min()
    img /= img.max()
    if mask is not None:
        img = (img + (1-mask[0,:,:,:].permute((1,2,0)))*100).clamp(0,1)
    plt.figure(figsize=(7,7))
    plt.imshow((img*brighten).clamp(0,1), origin="upper")
    
def polish_output(image, output, mask, blur=15):
    channel, width, height = image.size()
    mask = mask.cpu().numpy()[0,0,:,:]
    mask = cv2.blur(mask, (blur, blur))
    
    mask = mask - mask.min()
    mask = mask/mask.max()
    mask = torch.from_numpy(mask)
    _mean = image.view(channel,-1).mean(dim=1, keepdim=True)
    _std = image.view(channel,-1).std(dim=1, keepdim=True)
    output = output.view(channel, -1)
    output = (output - output.mean(1,keepdim=True))/output.std(1,keepdim=True)
    output = output * _std + _mean
    output = output.view(channel, width, height)
    comb = output*(1-mask) + image*mask
    comb = comb.view(channel, -1)
    comb = (comb - comb.mean(dim=1, keepdim=True))/comb.std(1, keepdim=True)
    comb = comb*_std + _mean
    comb = comb.view(channel, width, height)
    return comb