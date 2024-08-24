from torchvision import datasets, transforms
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split

custom_transform_train = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    transforms.RandomCrop(32, padding=4, padding_mode='reflect'),
    transforms.RandomHorizontalFlip()
])

custom_transform_valid = transforms.Compose([
    transforms.ToTensor(),
])


def apply_custom_transform(dataset, transform):
    dataset.transform = transform
    return dataset


def make_data_loader(args):
    # Obtenir le Dataset
    train_dataset = datasets.CIFAR10(args.data, train=True,  download=True)
    test_dataset = datasets.CIFAR10(args.data, train=False, transform=custom_transform_valid, download=True)

    validation_size = int(0.2* len(test_dataset))
    train_dataset, validation_dataset = random_split(test_dataset, [len(test_dataset) - validation_size, validation_size])

    train_dataset = apply_custom_transform(train_dataset, custom_transform_train)
    validation_dataset = apply_custom_transform(validation_dataset, custom_transform_train)

    # Obtenir le Dataloader
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    validation_loader = DataLoader(validation_dataset, batch_size=args.batch_size * 2)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False)
    
    return train_loader, validation_loader, test_loader

