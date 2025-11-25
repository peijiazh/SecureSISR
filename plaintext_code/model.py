import torch
import torch.nn as nn
from math import sqrt
from Relu_version import PolyReLU



class Conv_ReLU_Block(nn.Module):
    def __init__(self):
        super(Conv_ReLU_Block, self).__init__()
        self.conv = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1, bias=False)
        self.relu = PolyReLU()


    def forward(self, x):
        x = self.conv(x)
        x = self.relu(x)

        return x

class Network(nn.Module):
    def __init__(self):
        super(Network, self).__init__()

        self.residual_layer = self.make_layer(Conv_ReLU_Block, 7)
        # self.residual_layer_1 = self.make_layer(Conv_ReLU_Block, 2)
        # self.residual_layer_2 = self.make_layer(Conv_ReLU_Block, 2)

        self.input = nn.Conv2d(in_channels=3, out_channels=64,  kernel_size=3, stride=1, padding=1, bias=False)
        # self.output = nn.Conv2d(in_channels=64, out_channels=1,  kernel_size=3, stride=1, padding=1, bias=False)
        self.output = nn.Conv2d(in_channels=64, out_channels=3, kernel_size=3, stride=1, padding=1, bias=False)
        self.relu = PolyReLU()
        # self.pixel = nn.PixelShuffle(2)
        # self.upsample = torch.nn.ConvTranspose2d(in_channels=64, out_channels=64, kernel_size=8, stride=4, padding=2, bias=False)




        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, sqrt(2. / n))

    def make_layer(self, block, num_of_layer):
        layers = []
        for _ in range(num_of_layer):
            layers.append(block())
        return nn.Sequential(*layers)



    def forward(self, x):
        residual = x

        # residual = out
        inputs = self.relu(self.input(x))

        out = self.residual_layer(inputs)
        out = self.output(out)
        out = torch.add(out, residual)


        return out





if __name__ == '__main__':
    x = torch.randn(1,1,224,224)
    net = Network()
    print(net(x).shape)
