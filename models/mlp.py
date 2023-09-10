from __future__ import print_function
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

class MLP(nn.Module):
    ####for data4_1_1_1
    # def __init__(self, dx, num_classes):
    #     super(MLP, self).__init__()
    #     self.dx = dx
    #     self.fc1 = nn.Linear(self.dx, 3)
    #     # self.fc2 = nn.Linear(20, 20)
    #     # self.fc3 = nn.Linear(20, 20)
    #     # self.fc4 = nn.Linear(16, 4)
    #     self.fc5 = nn.Linear(3, num_classes)
    #     self.sigmoid = nn.Sigmoid()
    # def forward(self, x):
    #     out = F.relu(self.fc1(x.view(-1,self.dx)))
    #     # out = F.relu(self.fc2(out))
    #     # out = F.relu(self.fc3(out))
    #     # out = F.relu(self.fc4(out))
    #     out = self.fc5(out)
    #     out = self.sigmoid(out)
    #     return out

    # for data_200_imb_1
    # def __init__(self, dx, num_classes):
    #     super(MLP, self).__init__()
    #     self.dx = dx
    #     self.fc1 = nn.Linear(self.dx, 20)
    #     self.fc2 = nn.Linear(20, 20)
    #     self.fc3 = nn.Linear(20, 20)
    #     # self.fc4 = nn.Linear(16, 4)
    #     self.fc5 = nn.Linear(20, num_classes)
    #     self.sigmoid = nn.Sigmoid()
    # def forward(self, x):
    #     out = F.relu(self.fc1(x.view(-1,self.dx)))
    #     out = F.relu(self.fc2(out))
    #     out = F.relu(self.fc3(out))
    #     # out = F.relu(self.fc4(out))
    #     out = self.fc5(out)
    #     out = self.sigmoid(out)
    #     return out

    ####for data6_1
    def __init__(self, dx, num_classes):
        super(MLP, self).__init__()
        self.dx = dx
        self.fc1 = nn.Linear(self.dx, 64)
        # self.fc2 = nn.Linear(20, 20)
        # self.fc3 = nn.Linear(20, 20)
        # self.fc4 = nn.Linear(16, 4)
        self.fc5 = nn.Linear(64, num_classes)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        out = F.relu(self.fc1(x.view(-1, self.dx)))
        # out = F.relu(self.fc2(out))
        # out = F.relu(self.fc3(out))
        # out = F.relu(self.fc4(out))
        out = self.fc5(out)
        out = self.sigmoid(out)
        return out