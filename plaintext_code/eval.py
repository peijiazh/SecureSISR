import argparse, os
import torch
from torch.autograd import Variable
import numpy as np
import time, math, glob
import scipy.io as sio
import pytorch_ssim
import matlab.engine
from model import Network



parser = argparse.ArgumentParser(description="PyTorch VDSR Eval")
# parser.add_argument("--cuda", action="store_true", help="use cuda?")
parser.add_argument("--model", default="model_testrgb1smsrx2/model_epoch_50.pth", type=str, help="model path")
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



opt = parser.parse_args()
# cuda = opt.cuda
eng = matlab.engine.start_matlab()



model = torch.load(opt.model, map_location=lambda storage, loc: storage)["model"]



scales = [2, 3, 4]
# scales = [4]
image_list = glob.glob(opt.dataset + "_mat/*.*")


for scale in scales:
    avg_psnr_predicted = 0.0
    avg_psnr_bicubic = 0.0
    avg_ssim_predicted = 0.0
    avg_ssim_bicubic = 0.0
    avg_elapsed_time = 0.0
    count = 0.0

    for image_name in image_list:
        if str(scale) in image_name:
            count = count + 1
            im_gt_y = sio.loadmat(image_name)['im_gt_y']
            im_b_y = sio.loadmat(image_name)['im_b_y']
            im_b = sio.loadmat(image_name)['im_b']
            # im_l = sio.loadmat(image_name)['im_l']

            im_gt_y = im_gt_y.astype(float)
            im_b_y = im_b_y.astype(float)
            im_b = im_b.astype(float)
    

            psnr_bicubic = PSNR(im_gt_y, im_b_y, shave_border=scale)
            ssim_bicubic = SSIM(im_gt_y, im_b_y, shave_border=scale)

            avg_psnr_bicubic += psnr_bicubic
            avg_ssim_bicubic += ssim_bicubic


            im_input = im_b.astype(np.float32).transpose(2,0,1)

            im_input = Variable(torch.from_numpy(im_input/255.).float()).view(1, 3, im_input.shape[1], im_input.shape[2])


            
            model = model.cpu()

            start_time = time.time()
            HR= model(im_input)
            elapsed_time = time.time() - start_time
            avg_elapsed_time += elapsed_time


            HR = HR.cpu()

            im_h = HR.data[0].numpy().astype(np.float32)


            im_h = im_h * 255.
            im_h[im_h < 0] = 0
            im_h[im_h > 255.] = 255.
            im_h = im_h.transpose(1,2,0).astype(np.float32)

            im_h_matlab = matlab.double((im_h / 255.).tolist())
            im_h_ycbcr = eng.rgb2ycbcr(im_h_matlab)
            im_h_ycbcr = np.array(im_h_ycbcr._data).reshape(im_h_ycbcr.size, order='F').astype(np.float32) * 255.
            im_h_y = im_h_ycbcr[:, :, 0] 


            psnr_predicted = PSNR(im_gt_y, im_h_y, shave_border=scale)
            ssim_predicted = SSIM(im_gt_y, im_h_y, shave_border=scale)

            avg_psnr_predicted += psnr_predicted
            avg_ssim_predicted += ssim_predicted



    print("Scale=", scale)
    print("Dataset=", opt.dataset)
    print("PSNR_predicted=", avg_psnr_predicted / count)
    print("PSNR_bicubic=", avg_psnr_bicubic / count)
    print("SSIM_predicted=", avg_ssim_predicted / count)
    print("SSIM_bicubic=", avg_ssim_bicubic / count)
    # print("It takes average {}s for processing".format(avg_elapsed_time / count))
    # print("one time =", elapsed_time)


