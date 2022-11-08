import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
#%matplotlib inline

import torch
from torch.utils.data import DataLoader, Dataset
import torchvision
from torchvision import transforms

import albumentations as A
from albumentations.pytorch import ToTensor

# augmentation 파이프라인 선언, p는 확률을 의미
train_transform = A.Compose([
        A.RandomRotate90(),
        A.Flip(),
        A.Transpose(),
        A.OneOf([ #OneOf내에있는 기법들중에 랜덤으로 하나를 선택한다는 의미
            A.IAAAdditiveGaussianNoise(),
            A.GaussNoise(),
        ], p=0.2), # IAAAdditiveGaussianNoise와 GaussNoise 둘 중 하나의 기법만 사용하고 p값은 해당 OneOf가 실행 될 확률
        A.OneOf([
            A.MotionBlur(p=.2),
            A.MedianBlur(blur_limit=3, p=0.1),
            A.Blur(blur_limit=3, p=0.1),
        ], p=0.2),
        A.ShiftScaleRotate(shift_limit=0.0625, scale_limit=0.2, rotate_limit=45, p=0.2),
        A.OneOf([
            A.OpticalDistortion(p=0.3),
            A.GridDistortion(p=.1),
            A.IAAPiecewiseAffine(p=0.3),
        ], p=0.2),
        A.OneOf([
            A.CLAHE(clip_limit=2),
            A.IAASharpen(),
            A.IAAEmboss(),
            A.RandomBrightnessContrast(),
        ], p=0.3),
        A.HueSaturationValue(p=0.3),
    ToTensor()
    ])


class albumDataset(Dataset):
    def __init__(self, file_path, transform=None):
        self.data = pd.read_csv(file_path)
        self.transform = transform

    def __len__(self):
        return len(self.data)


    def __getitem__(self, index):
        img = self.data.iloc[index, 1:].values.astype(np.uint8).reshape((28, 28, 1))
        img = np.repeat(img, 3, 2)
        # np.repeat(img,3,2)는 데이터 shape (28,28,1) 을 (28,28,3)으로 만들어줌
        # grid를 쓰지않고 바로 결과를 보여주기 위함
        label = self.data.iloc[index, 0]

        if self.transform is not None:
            image = self.transform(image=img)['image']

        return image, label


file_path = "/home/hhs/PycharmProjects/pythonProject/csv/csv_image.csv"
train_dataset3 = albumDataset(file_path, transform = train_transform)

plt.imshow(train_dataset3[0][0].permute(1,2,0).numpy())