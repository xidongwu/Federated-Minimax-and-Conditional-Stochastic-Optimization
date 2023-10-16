# Cifar10
# https://towardsdatascience.com/cifar-10-image-classification-in-tensorflow-5b501f7dc77c
import torch
import torchvision
import torchvision.transforms as transforms
import numpy as np
import os
from PIL import Image
from collections import Counter
import torch.distributed as dist


transform_c = transforms.Compose(
    [transforms.ToTensor(),
     transforms.Normalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761))])


transform_m =transforms.Compose([
                               transforms.ToTensor(),
                               transforms.Normalize(
                                 (0.1307,), (0.3081,))
                             ])

transform_f = transforms.Compose([
    transforms.ToTensor(), 
    transforms.Normalize(
        (0.2860,), (0.3529,)) ])

transform_train_t = transforms.Compose([
            transforms.Resize(256), # Resize images to 256 x 256
            transforms.CenterCrop(224), # Center crop image
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),  # Converting cropped images to tensors
            transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                            std=[0.229, 0.224, 0.225])
])


transform_val_t = transforms.Compose([
            transforms.Resize(256), # Resize images to 256 x 256
            transforms.ToTensor(),  # Converting cropped images to tensors
            transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                            std=[0.229, 0.224, 0.225])
])


# 0 T-shirt/top; 1 Trouser; 2 Pullover; 3 Dress; 4 Coat; 
# 5 Sandal; 6 Shirt; 7 Sneaker; 8 Bag; 9 Ankle boot

class DISTFashionMNIST(torchvision.datasets.FashionMNIST):

    def __init__(self, root, rank, train=True,
                 transform=None, target_transform=None, download=False):
        super(DISTFashionMNIST, self).__init__(root, train, transform, target_transform, download)

        size = dist.get_world_size()

        new_data = []
        new_targets = []
        classes = np.unique(self.targets.numpy())
        targets_np = np.array(self.targets, dtype=np.int64)

        for class_ in range(10):
            idx = np.where(targets_np == class_)[0]
            new_data.append(self.data[idx, ...])
            new_targets.extend([class_, ] * len(idx))

        new_data = np.vstack(new_data)

        self.data = torch.from_numpy(new_data)[rank::size]
        self.targets = np.array(new_targets).tolist()[rank::size]

    def __getitem__(self, index: int):
        """
        Args:
            index (int): Index

        Returns:
            tuple: (image, target) where target is index of the target class.
        """
        img, target = super().__getitem__(index)
        return img, target


class DISTCIFAR10(torchvision.datasets.CIFAR10):

    def __init__(self, root, rank, train=True,
                 transform=None, target_transform=None, download=False):
        super(DISTCIFAR10, self).__init__(root, train, transform, target_transform, download)
        size = dist.get_world_size()

        new_data = []
        new_targets = []
        targets_np = np.array(self.targets, dtype=np.int64)
        classes = np.unique(targets_np)

        for class_ in range(10):
            idx = np.where(targets_np == class_)[0]
            new_data.append(self.data[idx, ...])
            new_targets.extend([class_, ] * len(idx))

        self.data = np.vstack(new_data)[rank::size]
        self.targets = np.array(new_targets).tolist()[rank::size]


    def __getitem__(self, index: int):
        """
        Args:
            index (int): Index

        Returns:
            tuple: (image, target) where target is index of the target class.
        """
        img, target = super().__getitem__(index)
        return img, target

# TINYIMAGENET
# https://towardsdatascience.com/pytorch-ignite-classifying-tiny-imagenet-with-efficientnet-e5b1768e5e8f

class TINYIMAGENET(torchvision.datasets.ImageFolder):
    cls_num = 200

    def __init__(self, root, train=False, transform=None):
        super(TINYIMAGENET, self).__init__(root, train, transform)
        rank = dist.get_rank()
        size = dist.get_world_size()
        # rank = 0
        # size = 1

        self.transform = transform

        img_max = int(len(self.imgs) / self.cls_num)

        new_data = []
        new_targets = []

        self.targets = [x[1] for x in self.imgs]
        self.imgs = np.array(self.imgs)

        targets_np = np.array(self.targets, dtype=np.int64)
        classes = np.unique(targets_np)

        for the_class in classes:
            idx = np.where(targets_np == the_class)[0]
            np.random.shuffle(idx)
            # selec_idx = idx[:the_img_num]
            new_data.extend(self.imgs[idx, ...])
            # label = the_class // 100
            new_targets.extend([the_class, ] * img_max)

        self.samples = new_data[rank::size]
        self.targets = new_targets[rank::size]

        # self.samples = np.array(self.imgs)

    def __getitem__(self, index):

        path, target = self.samples[index]
        target = int(target)
        assert os.path.isfile(path) == True, "File not exists"
        img = Image.open(path)
        sample = img.convert('RGB')

        if self.transform is not None:
            sample = self.transform(sample)
        # if self.target_transform is not None:
        #     target = self.target_transform(target)
        return sample, target


