import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
#% matplotlib inline

import torch
from torch.utils.data import DataLoader, Dataset
import torchvision
from torchvision import transforms


class Dataset(Dataset):

    def __init__(self, file_path, transform=None):
        self.data = pd.read_csv(file_path)
        self.transform = transform

        # __init__ : 데이터셋 전처리

    def __len__(self):
        return len(self.data)

        # __len__ : 데이터셋의 총 길이. 즉 총 데이터 수

    def __getitem__(self, index):
        image = self.data.iloc[index, 1:].values.astype(np.uint8).reshape((1, 28, 28))
        # type을 uint8로 해주는 이유는 이미지는 0~255의 값을 가지므로 256개를 가지는 unsigned integer로 지정
        # Pytorch 모델들은 (batchsize,width,height,channel) 이 아닌 (batchsize,channel,width,height)의 shape을 받음

        label = self.data.iloc[index, 0]

        if self.transform is not None:
            image = self.transform(image)

        return image, label

        # __getitem__ : 어떤 샘플을 가져올지 index를 받아서 그만큼 보내주는 함수


file_path = "/home/hhs/PycharmProjects/pythonProject/csv/csv_image_test.csv"
train_dataset = Dataset(file_path=file_path, transform=None)

image, label = train_dataset.__getitem__(0)
print(image.shape) #image의 shape은 (channel, width, height) 로 Pytorch에서 shape을 만들어주고 gray scale이기 때문에 channel은 1
print(type(image))

train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True)
#DataLoader의 첫번째 인자는 위에서 선언한 dataset을 받고 batch_size=8 이므로 8개의 데이터씩 뽑아낼 수 있음

train_iter = iter(train_loader)
#반복 가능한 객체를 iter라는 함수를 이용해 next로 계속해서 값을 받아올 수 있음

print(type(train_iter))

images, labels = train_iter.next()
print('images shape on batch size = {}'.format(images.size()))
print('labels shape on batch size = {}'.format(labels.size()))
#DataLoader에서 나온 데이터의 타입은 torch.Tensor

grid = torchvision.utils.make_grid(images)

print(grid.numpy().shape)
print(grid.numpy().transpose(1,2,0).shape)
print(grid.permute(1,2,0).numpy().shape)
#transpose말고 permute라는 함수가 있는데 기능은 transpose와 같음
#transpose는 numpy타입에 대해서만 사용하고(torch.Tensor형에서도 동작을 하지만 2개의 위치만 바꿀수 있기 때문에 width와 height을 바꿀때만 사용한다.) permute는 torch.Tensor 타입에서만 작동
