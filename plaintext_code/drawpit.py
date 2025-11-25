import argparse, os
import torch
from torch.autograd import Variable
import numpy as np
import time, math, glob
import scipy.io as sio
import pandas
import matplotlib.pyplot as plt
from PIL import Image



def read_tablemethod(filename):
    data = pandas.read_table(filename, header=None, delim_whitespace=True)
    return data


if __name__ == "__main__":
    data = read_tablemethod('./hr.txt')

    # data is DataFrame type, convert it to numpy array to display image
    img = data.values
    # img = img.numpy().astype(np.float32)
    img = img * 255.
    img[img < 0] = 0
    img[img > 255.] = 255.
    img = Image.fromarray(img)
    print(img)
    plt.imshow(img)
    plt.show()
