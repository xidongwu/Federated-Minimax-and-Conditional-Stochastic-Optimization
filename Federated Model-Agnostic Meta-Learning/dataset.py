# https://analyticsindiamag.com/guide-to-torchmeta-a-meta-learning-library-for-pytorch/
from torchmeta.datasets import Omniglot, MiniImagenet
from torchmeta.transforms import Categorical, ClassSplitter, Rotation
from torchvision.transforms import Compose, Resize, ToTensor
from torchmeta.utils.data import BatchMetaDataLoader
from    PIL import Image
import  os.path
import  numpy as np
import torch

import torch.distributed as dist

class SineWave(object):
    """
    """
    def __init__(self, batchsz, k_shot, rank):
        self.k_shot = k_shot
        self.k_qry = 10
        self.rank = rank
        self.A = self.rank + 1
        self.batchsz = batchsz
    
    def gen_task(self):

        data_x = np.zeros((self.batchsz, self.k_shot + self.k_qry,1))
        data_y = np.zeros((self.batchsz, self.k_shot + self.k_qry,1))

        task_code = []
        A = self.A
        for i in range(self.batchsz):
            Phi = np.random.randint(1, high=6)
            # A = 1; Phi = 1
            task_code.append(Phi - 1)

            data_x[i] = np.random.uniform(low=-5, high=5, size=self.k_shot + self.k_qry).reshape(1,-1,1)
            data_y[i] = A * np.sin(data_x[i] + Phi * np.pi/5)

        return (torch.tensor(data_x).cuda(), torch.tensor(data_y).cuda()), task_code

    def gen_all_tasks(self):
        data_x = np.zeros((5, self.k_shot + self.k_qry,1))
        data_y = np.zeros((5, self.k_shot + self.k_qry,1))

        task_code = []
        for j in range(5):
            Phi = j; task_code.append(Phi - 1)
            data_x[j] = np.random.uniform(low=-5, high=5, size=self.k_shot + self.k_qry).reshape(1,-1,1)
            data_y[j] = self.A * np.sin(data_x[j] + Phi * np.pi/5)

        return (torch.tensor(data_x).cuda(), torch.tensor(data_y).cuda()), task_code

    def gen_one_test_task(self):

        A = np.random.uniform(low = 1, high=5)
        Phi = np.random.uniform(low = 1, high=5)

        x = np.random.uniform(low=-5, high=5, size=self.k_shot + self.k_qry).reshape(1,-1,1)
        y = A * np.sin(x + Phi * np.pi/5)

        return torch.tensor(x).cuda(), torch.tensor(y).cuda()

    def gen_one_test_task_large(self):

        A = np.random.uniform(low = 1, high=5)
        Phi = np.random.uniform(low = 1, high=5)

        x = np.random.uniform(low=-5, high=5, size=2000).reshape(1,-1,1)
        y = A * np.sin(x + Phi * np.pi/5)

        return torch.tensor(x).cuda(), torch.tensor(y).cuda()


class OmniglotNShot:

    def __init__(self, batchsz, n_way, k_shot, k_query, rank, imgsz=28, size = 1):
        """
        Different from mnistNShot, the
        :param batchsz: task num
        :param n_way:
        :param k_shot:
        :param k_qry:
        :param imgsz:
        """

        self.resize = imgsz

        # if rank == 0:
        #     d_tag = True
        # else:
        #     d_tag = False
        # d_tag = True
        # print(rank, world)
        dataset = Omniglot("data",
                        # Number of ways
                        num_classes_per_task=n_way,
                        # Resize the images to 28x28 and converts them\
                        #  to PyTorch tensors (from Torchvision)
                        transform=Compose([Resize(imgsz), ToTensor()]),
                        # Transform the labels to integers (e.g.\
                        #  ("Glagolitic/character01", "Sanskrit/character14", ...) \
                        # to (0, 1, ...))
                        target_transform=Categorical(num_classes=n_way),
                        # Creates new virtual classes with rotated versions \
                        # of the images (from Santoro et al., 2016)
                        # class_augmentations=[Rotation([30, 90, 120, 180, 270])],
                        class_augmentations=[Rotation([90, 180, 270])],
                        meta_train=True, download=False, rank=rank, size=size)

        dist.barrier()
        dataset = ClassSplitter(dataset, shuffle=True, num_train_per_class=k_shot, \
            num_test_per_class=k_query)
        self.dataloader = BatchMetaDataLoader(dataset, batch_size=batchsz, num_workers=4)

        if rank == 0:
            dataset_val = Omniglot("data",
                            # Number of ways
                            num_classes_per_task=n_way,
                            # Resize the images to 28x28 and converts them\
                            #  to PyTorch tensors (from Torchvision)
                            transform=Compose([Resize(imgsz), ToTensor()]),
                            # Transform the labels to integers (e.g.\
                            #  ("Glagolitic/character01", "Sanskrit/character14", ...) \
                            # to (0, 1, ...))
                            target_transform=Categorical(num_classes=n_way),
                            # Creates new virtual classes with rotated versions \
                            # of the images (from Santoro et al., 2016)
                            # class_augmentations=[Rotation([30, 90, 120, 180, 270])],
                            class_augmentations=[Rotation([90, 180, 270])],
                            meta_val=True, download=False)

            dataset_val = ClassSplitter(dataset_val, shuffle=True, num_train_per_class=k_shot, \
                num_test_per_class=15)
            self.dataloader_val = BatchMetaDataLoader(dataset_val, shuffle=True, batch_size=batchsz, num_workers=4)
        # print(rank, 'I am here !!')
        dist.barrier()

class MiniImagenetNShot:

    def __init__(self, batchsz, n_way, k_shot, k_query, rank, imgsz=84, size = 1):
        """
        Different from mnistNShot, the
        :param batchsz: task num
        :param n_way:
        :param k_shot:
        :param k_qry:
        :param imgsz:
        """

        self.resize = imgsz

        # if rank == 0:
        #     d_tag = True
        # else:
        #     d_tag = False
        # d_tag = True
        # print(rank, world)
        dataset = MiniImagenet("data",
                        # Number of ways
                        num_classes_per_task=n_way,
                        # Resize the images to 28x28 and converts them\
                        #  to PyTorch tensors (from Torchvision)
                        transform=Compose([Resize(imgsz), ToTensor()]),
                        # Transform the labels to integers (e.g.\
                        #  ("Glagolitic/character01", "Sanskrit/character14", ...) \
                        # to (0, 1, ...))
                        target_transform=Categorical(num_classes=n_way),
                        # Creates new virtual classes with rotated versions \
                        # of the images (from Santoro et al., 2016)
                        # class_augmentations=[Rotation([30, 90, 120, 180, 270])],
                        class_augmentations=[Rotation([90, 180, 270])],
                        meta_train=True, download=False, rank= rank, size = size)

        dist.barrier()
        dataset = ClassSplitter(dataset, shuffle=True, num_train_per_class=k_shot, \
            num_test_per_class=k_query)
        self.dataloader = BatchMetaDataLoader(dataset, batch_size=batchsz, num_workers=4)

        if rank == 0:
            dataset_val = MiniImagenet("data",
                            # Number of ways
                            num_classes_per_task=n_way,
                            # Resize the images to 28x28 and converts them\
                            #  to PyTorch tensors (from Torchvision)
                            transform=Compose([Resize(imgsz), ToTensor()]),
                            # Transform the labels to integers (e.g.\
                            #  ("Glagolitic/character01", "Sanskrit/character14", ...) \
                            # to (0, 1, ...))
                            target_transform=Categorical(num_classes=n_way),
                            # Creates new virtual classes with rotated versions \
                            # of the images (from Santoro et al., 2016)
                            # class_augmentations=[Rotation([30, 90, 120, 180, 270])],
                            class_augmentations=[Rotation([90, 180, 270])],
                            meta_val=True, download=False)

            dataset_val = ClassSplitter(dataset_val, shuffle=True, num_train_per_class=k_shot, \
                num_test_per_class=15) # 15 for validation
            self.dataloader_val = BatchMetaDataLoader(dataset_val, shuffle=True, batch_size=batchsz, num_workers=4)
        # print(rank, 'I am here !!')
        dist.barrier()
if __name__ == '__main__':
    MiniImagenetNShot(4, 5, 1, 15, 0)
