'''LeNet in PyTorch.'''
import torch.nn as nn
import torch
import torch.nn.functional as F
from torch.autograd import Variable

class Fourcnn_mnist(nn.Module):
    def __init__(self, nc=1, nh=28, hw=28, num_classes=10):
        input_shape = (nc, nh, hw)
        super(Fourcnn_mnist, self).__init__()
        self.maxpool = nn.MaxPool2d((2, 2), stride=(2, 2))
        self.conv1 = nn.Conv2d(nc, 32, 3)
        self.batchnorm1_1 = nn.BatchNorm2d(32)
        self.conv2 = nn.Conv2d(32, 64, 3)
        self.batchnorm2_1 = nn.BatchNorm2d(64)
        self.flat_shape = self.get_flat_shape(input_shape)
        self.fc1 = nn.Linear(self.flat_shape, 128)
        self.batchnorm3_1 = nn.BatchNorm1d(128)
        self.fc2 = nn.Linear(128, num_classes)

    def get_flat_shape(self, input_shape):
        dummy = Variable(torch.zeros(1, *input_shape))
        dummy = self.maxpool(self.conv1(dummy))
        dummy = self.maxpool(self.conv2(dummy))
        return dummy.data.view(1, -1).size(1)

    def forward(self, x_in):
        # conv 1
        x = self.conv1(x_in)
        x = self.batchnorm1_1(x)
        x = F.relu(x)
        x = self.maxpool(x)

        # conv 2
        x = self.conv2(x)
        x = self.batchnorm2_1(x)
        x = F.relu(x)
        x = self.maxpool(x)

        x = x.view(-1, self.flat_shape)

        # fc 1
        x = self.fc1(x)
        x = self.batchnorm3_1(x)
        x = F.relu(x)

        # fc 2
        x_out1 = self.fc2(x)

        return x_out1

