import torch
import random
import math
import numpy as np
from numba import jit
import torch.nn as nn
import torch.nn.functional as F
from easydict import EasyDict as edict
from utils.filters import ExposureFilter, DefogFilter,ImprovedWhiteBalanceFilter,GammaFilter, ToneFilter, ContrastFilter, UsmFilter

def conv_downsample(in_filters, out_filters, normalization=False):
    layers = [nn.Conv2d(in_filters, out_filters, 3, stride=2, padding=1)]
    layers.append(nn.LeakyReLU(0.2))
    if normalization:
        layers.append(nn.InstanceNorm2d(out_filters, affine=True))
    return layers


# class Dip_Cnn(nn.Module):
#     def __init__(self):
#         super(Dip_Cnn, self).__init__()
#         channels = 16
#         self.cnnnet = nn.Sequential(
#                         # 3, 16, 3, 2, 1
#                         nn.Conv2d(3, channels, kernel_size=3, stride=2, padding=1, bias=True),
#                         nn.LeakyReLU(negative_slope=0.1),
                        
#                         # 16, 32, 3, 2, 1
#                         nn.Conv2d(channels, channels*2, kernel_size=3, stride=2, padding=1, bias=True),
#                         nn.LeakyReLU(negative_slope=0.1),
                        
#                         # 32, 32, 3, 2, 1
#                         nn.Conv2d(channels*2, channels*2, kernel_size=3, stride=2, padding=1, bias=True),
#                         nn.LeakyReLU(negative_slope=0.1),
                        
#                         # 32, 32, 3, 2, 1
#                         nn.Conv2d(channels*2, channels*2, kernel_size=3, stride=2, padding=1, bias=True),
#                         nn.LeakyReLU(negative_slope=0.1),
                        
#                         # 32, 32, 3, 2, 1
#                         nn.Conv2d(channels*2, channels*2, kernel_size=3, stride=2, padding=1, bias=True),
#                         nn.LeakyReLU(negative_slope=0.1),
#         )
#         self.full_layers1 = nn.Sequential(
#                         nn.Linear(2048, 64),
#                         nn.Linear(64, 15),
#         )
#     def forward(self, x):
#         # resized_data = F.interpolate(x, size=(256, 256), mode='bilinear', align_corners=False)
#         out = self.cnnnet(x) # B, 3, 256, 256 ---> B, 32, 8, 8
#         out = out.reshape(out.size(0), -1) # B, 2048
#         out = self.full_layers1(out) # B, 15 #参数
#         return out

class Dip_Cnn(nn.Module):
    def __init__(self):
        super(Dip_Cnn, self).__init__()
        channels = 16
        self.cnnnet = nn.Sequential(
                        
                        # 3, 16, 3, 2, 1
                        nn.Conv2d(3, channels, kernel_size=3, stride=2, padding=1, bias=True),
                        nn.LeakyReLU(negative_slope=0.1),
                        nn.InstanceNorm2d(16, affine=True),
                        # 16, 32, 3, 2, 1
                        *conv_downsample(16, 32, normalization=True),
                        *conv_downsample(32, 64, normalization=True),
                        *conv_downsample(64, 128, normalization=True),
                        *conv_downsample(128, 128),
                        nn.Dropout(p=0.5),
                        nn.Conv2d(128, cfg.num_filter_parameters, 8, padding=0),
        )
        # self.full_layers1 = nn.Sequential(
        #                 nn.Linear(2048, 64),
        #                 nn.Linear(64, 15),
        # )
    def forward(self, x):
        # resized_data = F.interpolate(x, size=(256, 256), mode='bilinear', align_corners=False)
        out = self.cnnnet(x) # B, 3, 256, 256 ---> B, 32, 8, 8
        # out = out.reshape(out.size(0), -1) # B, 2048
        # out = self.full_layers1(out) # B, 15 #参数
        # import pdb
        # pdb.set_trace()
        out = out.reshape(out.size(0), -1) # B, 2048
        return out


# # def fog_image(image):
# #     @jit
# #     def AddHaz_loop(img_f, center, size, beta, A):
# #         (row, col, chs) = img_f.shape
# #         for j in range(row):
# #             for l in range(col):
# #                 d = -0.04 * math.sqrt((j - center[0]) ** 2 + (l - center[1]) ** 2) + size
# #                 td = math.exp(-beta * d)
# #                 img_f[j][l][:] = img_f[j][l][:] * td + A * (1 - td)
# #         return img_f
# #     img_f = image/255
# #     (row, col, chs) = image.shape
# #     A = 0.5  
# #     # beta = 0.08 
# #     beta = random.randint(0, 9) 
# #     beta = 0.01 * beta + 0.05
# #     size = math.sqrt(max(row, col)) 
# #     center = (row // 2, col // 2)  
# #     foggy_image = AddHaz_loop(img_f, center, size, beta, A)
# #     img_f = np.clip(foggy_image*255, 0, 255)
# #     img_f = img_f.astype(np.uint8)

# #     return img_f
# #----------------
# #过滤器参数
# #---------------
cfg = edict() 
# cfg.filters = [
#     DefogFilter, ImprovedWhiteBalanceFilter,  GammaFilter,
#     ToneFilter, ContrastFilter, UsmFilter
# ]
# cfg.num_filter_parameters = 15
# cfg.defog_begin_param = 0
# cfg.wb_begin_param = 1
# cfg.gamma_begin_param = 4
# cfg.tone_begin_param = 5
# cfg.contrast_begin_param = 13
# cfg.usm_begin_param = 14


# cfg.filters = [
#     ExposureFilter, DefogFilter, ImprovedWhiteBalanceFilter,  GammaFilter,
#     ToneFilter, ContrastFilter, UsmFilter
# ]
# cfg.num_filter_parameters = 16
# cfg.defog_begin_param = 0
# cfg.wb_begin_param = 1
# cfg.gamma_begin_param = 4
# cfg.tone_begin_param = 5
# cfg.contrast_begin_param = 13
# cfg.usm_begin_param = 14
# cfg.exposure_begin_param = 15


cfg.filters = [
    ImprovedWhiteBalanceFilter,  GammaFilter,
    ToneFilter, ContrastFilter, UsmFilter, ExposureFilter
]
cfg.num_filter_parameters = 15
# cfg.defog_begin_param = 0
cfg.wb_begin_param = 0
cfg.gamma_begin_param = 3
cfg.tone_begin_param = 4
cfg.contrast_begin_param = 12
cfg.usm_begin_param = 13
cfg.exposure_begin_param = 14


cfg.curve_steps = 8
cfg.gamma_range = 3
cfg.exposure_range = (-3.5, 3.5)
cfg.wb_range = 1.1
cfg.color_curve_range = (0.90, 1.10)
cfg.lab_curve_range = (0.90, 1.10)
cfg.tone_curve_range = (0.5, 2)
# cfg.defog_range = (0.1, 1.0)
cfg.defog_range = (0.1, 1.0)
cfg.usm_range = (0.0, 5) 
cfg.cont_range = (0.0, 1.0)       

         

# cfg.filters = [
#     DefogFilter, ExposureFilter, ImprovedWhiteBalanceFilter, SaturationPlusFilter,
#     GammaFilter, ToneFilter, ContrastFilter, UsmFilter
# ]
# cfg.num_filter_parameters = 17
#
# cfg.defog_begin_param = 0
# cfg.exposure_begin_param = 1
# cfg.wb_begin_param = 2
# cfg.saturation_begin_param = 5
# cfg.gamma_begin_param = 6
# cfg.tone_begin_param = 7
# cfg.contrast_begin_param = 15
# cfg.usm_begin_param = 16



def Dip_Filters(features, cfg, img):
    filtered_image_batch = img
    B, C, W, H = img.shape # [b, 3, 1330, 1330]
    #----------------
    #构建defog的过滤器参数
    #----------------
    dark = torch.zeros([B, W, H],dtype=torch.float32).to(img.device) # torch.Size([4, 640, 640]) 
    defog_A = torch.zeros([B, C],dtype=torch.float32).to(img.device) # [b, 3] 
    IcA = torch.zeros([B, W, H],dtype=torch.float32).to(img.device)  # torch.Size([4, 640, 640])
    for i in range(B):
        dark_i = DarkChannel(img[i]) # torch.Size([640, 640])
        defog_A_i = AtmLight(img[i], dark_i) # torch.Size([1, 3])
        IcA_i = DarkIcA(img[i], defog_A_i)  # torch.Size([640, 640])
        dark[i, ...] = dark_i
        defog_A[i, ...] = defog_A_i # [1, 3]
        IcA[i, ...] = IcA_i
    IcA = IcA.unsqueeze(-1) # torch.Size([4, 640, 640, 1])
    #需要经过的6个过滤器
    filters = cfg.filters
    filters = [x(filtered_image_batch, cfg) for x in filters]

    filter_parameters = []
    filtered_images = []

    filter_features = features # [B ,15]
    for j, filter in enumerate(filters):

        filtered_image_batch, filter_parameter = filter.apply(filtered_image_batch, filter_features, defog_A, IcA)
        filter_parameters.append(filter_parameter)
        filtered_images.append(filtered_image_batch)
    
    return filtered_image_batch

def DarkChannel(im):
    R = im[0, :, :] # [w,h]
    G = im[1, :, :] # [w,h]
    B = im[2, :, :] # [w,h]
    dc = torch.min(torch.min(R, G), B) # [w,h]
    return dc

def AtmLight(im, dark):
    c, h, w = im.shape # [c, h, w]
    imsz = h * w
    numpx = int(max(torch.floor(torch.tensor(imsz) / 1000), torch.tensor(1))) # h * w / 1000 取整
    darkvec = dark.reshape(imsz, 1) # [w*h, 1]
    imvec = im.reshape(3, imsz) # [c, w*h]

    indices = torch.argsort(darkvec) # [w*h, 1]
    indices = indices[(imsz - numpx):imsz]

    atmsum = torch.zeros([3, 1]).to(imvec.device)
    for ind in range(1, numpx):
        atmsum = atmsum + imvec[:, indices[ind]]

    A = atmsum / numpx # [3 ,1]
    return A.reshape(1, 3)

def DarkIcA(im, A):
    c, h, w = im.shape
    im3 = torch.zeros([c,h,w]).to(im.device)
    for ind in range(0, 3):
        im3[ind, :, :] = im[ind, :, :] / A[0, ind]
    return DarkChannel(im3)


