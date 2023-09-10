from __future__ import print_function
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

class Linear(nn.Module):
    def __init__(self, dx, num_classes):
        super(Linear, self).__init__()
        self.dx = dx
        self.fc = nn.Linear(self.dx, num_classes)
        self.sigmoid = nn.Sigmoid()
    def forward(self, x):
        out = self.fc(x.view(-1,self.dx))
        out = self.sigmoid(out)
        return out