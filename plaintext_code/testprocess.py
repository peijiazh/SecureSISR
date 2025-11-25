
import scipy.io as sio

import numpy as np
import torch
from torch.autograd import Variable
import argparse, os
import torch
from torch.autograd import Variable
import numpy as np
import time, math, glob
import scipy.io as sio
import pytorch_ssim
import matlab.engine

import time, math


parser = argparse.ArgumentParser(description="PyTorch VDSR Eval")
# parser.add_argument("--cuda", action="store_true", help="use cuda?")
parser.add_argument("--model", default="model_testrgb/model_epoch_50.pth", type=str, help="model path")
parser.add_argument("--dataset", default="data/test_data/Set5_rgb", type=str, help="dataset name, Default: Set5")
parser.add_argument("--gpus", default="", type=str, help="gpu ids (default: 0)")

def PSNR(pred, gt, shave_border=0):
    height, width = pred.shape[:2]
    pred = pred[shave_border:height - shave_border, shave_border:width - shave_border]
    gt = gt[shave_border:height - shave_border, shave_border:width - shave_border]
    imdff = pred - gt
    rmse = math.sqrt(np.mean(imdff ** 2))

    if rmse == 0:
        return 100
    return 20 * math.log10(255.0 / rmse)


def SSIM(pred, gt, shave_border=0, msg=""):
    height, width = pred.shape[:2]
    pred = pred[shave_border:height - shave_border, shave_border:width - shave_border]/255.0
    gt = gt[shave_border:height - shave_border, shave_border:width - shave_border]/255.0

    pred = torch.Tensor(pred, )
    pred = pred[None, None, :, :]
    gt = torch.Tensor(gt)
    gt = gt[None, None, :, :]

    if torch.cuda.is_available():
        pred = pred.cuda()
        gt = gt.cuda()

    # print("ret is ",pytorch_ssim.ssim(pred, gt).cpu().numpy())
    return pytorch_ssim.ssim(pred, gt).cpu().numpy()

def computeBic(src, srcH,  srcW, dstH, dstW):
    initdst = np.zeros((dstH,dstW,3))
    for i in range(0, dstH):
        for j in range(0, dstW):
            posx = i * srcH / dstH
            posy = j * srcW / dstW
            posxint = int(posx)
            posyint = int(posy)
            u = posx - posxint
            v = posy - posyint
            wx = get_wx(u)
            wy = get_wx(v)
            tmp = np.zeros(3)
            for ii in range(-1, 3):
                for jj in range(-1, 3):
                    if posxint+ii>=0 and posyint+jj>=0 and posxint+ii<srcH and posyint+jj<srcW:
                        w = wx[ii+1] * wy[jj+1]
                        tmp = tmp + src[posxint+ii,posyint+jj,:] * w
            initdst[i,j,:] = tmp
    return initdst


def get_wx(u):
    a = -0.5
    x = np.zeros(4)
    wx = np.zeros(4)
    x[0] = 1 + u
    x[1] = u
    x[2] = 1 - u
    x[3] = 2 - u
    wx[0] = a * abs(pow(x[0], 3)) -5 * a * abs(pow(x[0], 2)) + 8 * a * abs(x[0]) - 4 * a
    wx[1] = (a+2) * abs(pow(x[1], 3)) - (a+3) * abs(pow(x[1], 2)) + 1
    wx[2] = (a+2) * abs(pow(x[2], 3)) - (a+3) * abs(pow(x[2], 2)) + 1
    wx[3] = a * abs(pow(x[3], 3)) -5 * a * abs(pow(x[3], 2)) + 8 * a * abs(x[3]) - 4 * a
    return wx
#
# scale = 2
# weight = []
# for m in range(scale-1,-1,-1):
#     for n in range(scale-1,-1,-1):
#         u = m * (1/scale)
#         v = n * (1/scale)
#         wx = get_wx(u)
#         wy = get_wx(v)
#         W = np.zeros((4,4))
#         for i in range(0,4):
#             for j in range(0,4):
#                 W[i,j] = wx[i] * wy[j]
#         W = torch.tensor(W)
#         W = W.unsqueeze(dim=0)
#         weight.append(W)
# res = torch.cat(weight,0)
# res = res.unsqueeze(dim=0)
# res = F.pixel_shuffle(res, upscale_factor=scale)
# res = res.squeeze()
# res = res.detach().numpy()
# print(res)
#
# np.savetxt('Bicweight.txt',res)

# Three original image arrangement methods for 32*32
# image_name = 'C:/Users/87834/Desktop/testpit/Set5.mat'
# im_l = sio.loadmat(image_name)['im_l']
# im_l = im_l.astype(float)
# sizeRow, sizeCol, c = im_l.shape
# scale = 2
# dstH = sizeRow * scale
# dstW = sizeCol * scale
# zeros = np.zeros((sizeRow,sizeCol))
# res = im_l[:,:,0] /255.
# print(res.shape)
# res = torch.tensor(res)
# res = res.unsqueeze(dim=0)
# zeros = torch.tensor(zeros)
# zeros = zeros.unsqueeze(dim=0)
# res = torch.cat((res,zeros,zeros,zeros),0)
# res = res.unsqueeze(dim=0)
# res = F.pixel_shuffle(res, upscale_factor=2)
# res = res.squeeze()
# res = res.detach().numpy()
# np.savetxt('pitr',res)
# print(res)



############ plaintext computation ########################################
opt = parser.parse_args()
eng = matlab.engine.start_matlab()
model = torch.load(opt.model, map_location=lambda storage, loc: storage)["model"]
image_name = 'C:/Users/87834/Desktop/testpit/Set5.mat'
# im_b = sio.loadmat(image_name)['im_b']
# im_b = im_b.astype(float)
im_gt= sio.loadmat(image_name)['image']
im_gt = im_gt.astype(float)
im_l= sio.loadmat(image_name)['im_l']
im_l = im_l.astype(float)/255.
im_b = computeBic(im_l, 128, 128,256,256)




###################### ciphertext computation ###############################################
import numpy as np
def openreadtxt(file_name):
    data = []
    file = open(file_name,'r')  
    file_data = file.readlines() 
    for row in file_data:
        tmp_list = row.split(' ') 
        tmp_list[-1] = tmp_list[-1].replace('\n','') 
        tmp = np.array(tmp_list)
        tmp = tmp.astype(np.float32)
        data.append(tmp) 
    return data
data = openreadtxt('pitres.txt')
data = np.asarray(data)
data = data.astype(np.float32)
data = data.reshape((3, 256, 256))
# print(data.shape)
data = data * 255.
data[data < 0] = 0
data[data> 255.] = 255.

im_h = data.transpose(1,2,0)

im_h_matlab = matlab.double((im_h / 255.).tolist())
im_h_ycbcr = eng.rgb2ycbcr(im_h_matlab)
im_h_ycbcr = np.array(im_h_ycbcr._data).reshape(im_h_ycbcr.size, order='F').astype(np.float32) * 255.
im_h_y = im_h_ycbcr[:, :, 0]

im_gt_matlab = matlab.double((im_gt / 255.).tolist())
im_gt_ycbcr = eng.rgb2ycbcr(im_gt_matlab)
im_gt_ycbcr = np.array(im_gt_ycbcr._data).reshape(im_gt_ycbcr.size, order='F').astype(np.float32) * 255.
im_gt_y = im_gt_ycbcr[:, :, 0]


psnr_predicted = PSNR(im_gt_y, im_h_y, shave_border=2)
ssim_predicted = SSIM(im_gt_y, im_h_y, shave_border=2)
print(psnr_predicted)
print(ssim_predicted)