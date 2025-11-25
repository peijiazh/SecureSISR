import argparse, os
import torch
from torch.autograd import Variable
import numpy as np
import time, math, glob
import scipy.io as sio
import pytorch_ssim
from model import Network

parser = argparse.ArgumentParser(description="PyTorch VDSR Eval")
parser.add_argument("--cuda", action="store_true", help="use cuda?")
parser.add_argument("--model", default="model/model_epoch_50.pth", type=str, help="model path")
parser.add_argument("--dataset", default="data/test_data/Set5", type=str, help="dataset name, Default: Set5")
parser.add_argument("--gpus", default="0", type=str, help="gpu ids (default: 0)")



def PSNR(pred, gt, shave_border=0):
    height, width = pred.shape[:2]
    pred = pred[shave_border:height - shave_border, shave_border:width - shave_border]
    gt = gt[shave_border:height - shave_border, shave_border:width - shave_border]
    imdff = pred - gt
    rmse = math.sqrt(np.mean(imdff ** 2))
    if rmse == 0:
        return 100
    return 20 * math.log10(255.0 / rmse)


def SSIM(pred,gt,shave_border=0,msg=""):
    height, width = pred.shape[:2]
    pred = pred[shave_border:height - shave_border, shave_border:width - shave_border]/255.0
    gt = gt[shave_border:height - shave_border, shave_border:width - shave_border]/255.0

    pred = torch.Tensor(pred,)
    pred = pred[None,None,:,:]
    gt = torch.Tensor(gt)
    gt = gt[None,None,:,:]

    if torch.cuda.is_available():
        pred = pred.cuda()
        gt = gt.cuda()

    # print("ret is ",pytorch_ssim.ssim(pred, gt).cpu().numpy())
    return pytorch_ssim.ssim(pred, gt).cpu().numpy()


opt = parser.parse_args()
cuda = opt.cuda

if cuda:
    print("=> use gpu id: '{}'".format(opt.gpus))
    os.environ["CUDA_VISIBLE_DEVICES"] = opt.gpus
    if not torch.cuda.is_available():
        raise Exception("No GPU found or Wrong gpu id, please run without --cuda")

model = torch.load(opt.model, map_location=lambda storage, loc: storage)["model"]


image_name = 'test_picture/img087.mat'
scale = 4

im_gt_y = sio.loadmat(image_name)['im_gt_y']

im_b_y = sio.loadmat(image_name)['im_b_y']
im_bn_y = sio.loadmat(image_name)['im_bn_y']

im_gt_y = im_gt_y.astype(float)
im_b_y = im_b_y.astype(float)
im_bn_y = im_bn_y.astype(float)

psnr_bicubic = PSNR(im_gt_y, im_b_y, shave_border=scale)
ssim_bicubic = SSIM(im_gt_y, im_b_y, shave_border=scale)

psnr_bilnear = PSNR(im_gt_y, im_bn_y, shave_border=scale)
ssim_bilnear = SSIM(im_gt_y, im_bn_y, shave_border=scale)

im_input = im_b_y / 255.

im_input = Variable(torch.from_numpy(im_input).float()).view(1, -1, im_input.shape[0],
                                                                                im_input.shape[1])

if cuda:
    model = model.cuda()
    im_input = im_input.cuda()
else:
    model = model.cpu()

HR = model(im_input)

HR = HR.cpu()

im_h_y = HR.data[0].numpy().astype(np.float32)

im_h_y = im_h_y * 255.
im_h_y[im_h_y < 0] = 0
im_h_y[im_h_y > 255.] = 255.
im_h_y = im_h_y[0, :, :]

psnr_predicted = PSNR(im_gt_y, im_h_y, shave_border=scale)
ssim_predicted = SSIM(im_gt_y, im_h_y, shave_border=scale)


print("PSNR_predicted=", psnr_predicted )
print("PSNR_bicubic=", psnr_bicubic )
print("PSNR_bilnear=", psnr_bilnear )
print("SSIM_predicted=", ssim_predicted )
print("SSIM_bicubic=", ssim_bicubic )
print("ssim_bilnear=", ssim_bilnear)


