import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
#% matplotlib inline

import torch
from torch.utils.data import DataLoader, Dataset
import torchvision
from torchvision import transforms


class DatasetMNIST2(Dataset):

    def __init__(self, file_path, transform=None):
        self.data = pd.read_csv(file_path)
        self.transform = transform
        # transform에서는 ToTensor()를 사용
        # ToTensor는 (width,height,channel) shape을 (channel,width,height)로 바꿔주고 0~255의 픽셀값을 0~1값으로 normalize해줌

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        image = self.data.iloc[index, 1:].values.astype(np.uint8).reshape((28, 28, 1))
        # ToTensor를 쓰기 위해서는 (width,height,channel) shape을 가지고 있어야 함
        label = self.data.iloc[index, 0]

        if self.transform is not None:
            image = self.transform(image)

        return image, label


file_path = "/home/hhs/PycharmProjects/pythonProject/csv/csv_image_test.csv"
train_dataset2 = DatasetMNIST2(file_path=file_path, transform=torchvision.transforms.ToTensor())

img,lab = train_dataset2.__getitem__(0)
print('image shape at the first row : {}'.format(img.size()))

train_dataloader2 = DataLoader(train_dataset2, batch_size=8, shuffle=True)

train_iter2 = iter(train_dataloader2)
images, labels = train_iter2.next()

grid = torchvision.utils.make_grid(images)

plt.imshow(grid.permute(1,2,0).numpy())
plt.axis('off')
plt.title(labels.numpy())