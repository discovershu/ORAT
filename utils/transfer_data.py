# from PIL import Image
from torch.utils.data import Dataset
from torchvision.transforms import transforms
import numpy as np
import torch
import scipy.io
from sklearn.preprocessing import MinMaxScaler


# __all__ = ['MyDataset']

def get_train_val_test_split_mydata(seed, Toydata, train_size_precent):
    data = Toydata['x']
    data = MinMaxScaler(feature_range=(0, 1)).fit_transform(data)
    label = Toydata['y']
    uniqueValues, occurCount = np.unique(Toydata['y'], return_counts=True)
    number_1 = uniqueValues[0]
    number_2 = uniqueValues[1]

    n_samples = data.shape[0]

    number_1_index = np.where((label == number_1)== True)[0]
    number_2_index = np.where((label == number_2) == True)[0]

    remaining_indices = list(range(n_samples))

    random_state = np.random.RandomState(seed)

    train_size = train_size_precent

    train_indices_number_1_index = random_state.choice(number_1_index, int(number_1_index.size*train_size), replace=False)
    train_indices_number_2_index = random_state.choice(number_2_index, int(number_2_index.size * train_size), replace=False)
    train_indices = np.concatenate((train_indices_number_1_index, train_indices_number_2_index), axis=None)
    test_indices = np.setdiff1d(remaining_indices, train_indices)

    X_train = data[train_indices, :]
    X_train = np.float32(X_train)
    y_train = label[train_indices, :].ravel()

    X_test = data[test_indices, :]
    X_test = np.float32(X_test)
    y_test = label[test_indices, :].ravel()

    uniqueValues, occurCount = np.unique(label, return_counts=True)
    number_1 = uniqueValues[0]
    number_2 = uniqueValues[1]
    print('We do binary classificaiton using number: ', number_1, 'and', number_2)

    Y_train = [0 if x == number_1 else 1 for x in y_train]
    Y_train = np.asarray(Y_train)
    Y_test = [0 if x == number_1 else 1 for x in y_test]
    Y_test = np.asarray(Y_test)

    return X_train, Y_train, X_test, Y_test

class MyDataset(Dataset):

    def __init__(self,x,y, transform=None, target_transform=None):
        data = []
        for i in range(len(x)):
            data.append((x[i],y[i]))
        self.data = data
        self.transform = transform
        self.target_transform = target_transform

    def __getitem__(self, index):

        feature, label = self.data[index]
        return feature, label

    def __len__(self):
        return len(self.data)


def train_test_sets_generation(seed, Toydata_path, train_size_precent):

    # train_size_precent=0.5
    # Toydata_path = '/media/Data/shu/qSGD/dataset/mydata/data1.mat'
    Toydata = scipy.io.loadmat(Toydata_path)
    X_train, y_train, X_test, y_test = get_train_val_test_split_mydata(seed, Toydata, train_size_precent)
    transform_train = None
    trainset = MyDataset(X_train, y_train, transform=transform_train)
    testset = MyDataset(X_test, y_test, transform=transform_train)
    # trainloader = torch.utils.data.DataLoader(trainset, batch_size=8, shuffle=True)
    # testloader = torch.utils.data.DataLoader(testset, batch_size=8, shuffle=True)
    return trainset, testset
