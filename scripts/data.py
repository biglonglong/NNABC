# from torchvision.datasets import FashionMNIST
from torchvision import transforms
from torchvision.datasets import ImageFolder
import torch.utils.data as Data
import numpy as np


def train_val_data_process():
    # train_data = FashionMNIST(root='./data', 
    #                       train = True,
    #                       transform=transforms.Compose([transforms.Resize(size=28), transforms.ToTensor()]),    # mark
    #                       download=True)

    ROOT_TRAIN = r'.\dataset_split\train'

    normalize = transforms.Normalize([0.162, 0.151, 0.138], [0.058, 0.052, 0.048])
    transform_train = transforms.Compose([transforms.Resize((224,224)), transforms.ToTensor(), normalize])

    train_data = ImageFolder(ROOT_TRAIN, transform_train)

    class_label = train_data.classes
    class_idx = train_data.class_to_idx

    train_data, val_data = Data.random_split(train_data, lengths=[round(0.8*len(train_data)), round(0.2*len(train_data))])

    train_dataloader = Data.DataLoader(dataset=train_data,
                                       batch_size=64,  # mark
                                       shuffle=True,
                                       num_workers=2)
    val_dataloader = Data.DataLoader(dataset=val_data,
                                       batch_size=64,  # mark
                                       shuffle=True,
                                       num_workers=2)

    return train_dataloader, val_dataloader


if __name__ == '__main__':
    '''[batch_size, channel_size, height, weight]'''
    train_val_data_process()