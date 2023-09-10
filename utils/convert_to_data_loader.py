# from PIL import Image
from torch.utils.data import Dataset
from torchvision.transforms import transforms
import numpy as np
import torch
import scipy.io
from PIL import Image

class MyDataset(Dataset):

    def __init__(self,x,y, transform=None, target_transform=None):
        data = []
        x = x.transpose((0, 2, 3, 1))
        for i in range(len(x)):
            data.append((x[i],y[i]))
        self.data = data
        self.transform = transform
        self.target_transform = target_transform


    def __getitem__(self, index):

        feature, label = self.data[index]
        # feature = Image.fromarray(feature.astype(np.uint8))
        if self.transform is not None:
            feature = self.transform(feature)
        return feature, label

    def __len__(self):
        return len(self.data)

def dataloader_generation(data_path):
    transform_train = transforms.Compose([transforms.ToTensor()])
    transform_test = transforms.Compose([transforms.ToTensor()])
    data = scipy.io.loadmat(data_path)
    X_train = data['X_train']
    y_train = np.int64(data['Y_train'][0])
    X_test = data['X_test']
    y_test = np.int64(data['Y_test'][0])
    trainset = MyDataset(X_train, y_train, transform=transform_train)
    testset = MyDataset(X_test, y_test, transform=transform_test)
    return trainset, testset
