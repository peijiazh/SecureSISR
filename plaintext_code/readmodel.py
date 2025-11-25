import argparse, os
import torch
from torch.autograd import Variable
import numpy as np



parser = argparse.ArgumentParser(description="PyTorch VDSR Eval")
parser.add_argument("--cuda", action="store_true", help="use cuda?")
parser.add_argument("--model", default="model_testrgb/model_epoch_50.pth", type=str, help="model path")
parser.add_argument("--gpus", default="", type=str, help="gpu ids (default: 0)")


opt = parser.parse_args()
cuda = opt.cuda

if cuda:
    print("=> use gpu id: '{}'".format(opt.gpus))
    os.environ["CUDA_VISIBLE_DEVICES"] = opt.gpus
    if not torch.cuda.is_available():
        raise Exception("No GPU found or Wrong gpu id, please run without --cuda")

model = torch.load(opt.model, map_location=lambda storage, loc: storage)["model"]

counter = 0

def save_text(x,pwd,name):

    global counter
    tmp_x= x.cpu().detach().numpy()
    tmp_x =  tmp_x.reshape((-1,tmp_x.shape[-1]))
    np.savetxt(pwd+str(counter)+name+'.txt',tmp_x)
    print('saving to '+pwd+str(counter)+name+'.txt')
    counter = counter + 1


path = "weight/"
if not os.path.exists(path):
    os.makedirs(path)
for name, param in model.named_parameters():
    save_text(param, path, name)
    print(name)






