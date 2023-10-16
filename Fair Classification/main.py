# CUDA_VISIBLE_DEVICES=1,2 python main.py --dataset mnist --method fedavg --worker-size 1 --epochs 10 --batch-size 5 --lr 0.005
import os
import torch
import torch.distributed as dist
from torch.multiprocessing import Process
from torchvision import datasets, transforms
import torch.optim as optim
from net import *
from dist_data import *
from SGDA import *
from FSGDA import *
import numpy as np
import argparse
import time

def get_default_device(idx):
    # print(idx)
    if torch.cuda.is_available():
        print("GPU available")
        return torch.device('cuda:' + str(int(idx)))
    else:
        print("GPU NOT available")
        return torch.device('cpu')
    # return torch.device('cpu')

""" Partitioning Dataset """
def partition_dataset(args):

    rank = dist.get_rank()
    size = dist.get_world_size()
    
    if args.dataset == 'fmnist':
        dataset = DISTFashionMNIST(root='../../data/', rank=rank, train=True, download=False, transform=transform_f)
        testset = DISTFashionMNIST(root='../../data/', rank=rank, train=False, download=False, transform=transform_f)
    elif args.dataset == 'cifar10':
        dataset = DISTCIFAR10('../../data/cifar10/', rank=rank, train=True, download=False, transform=transform_c)
        testset = DISTCIFAR10('../../data/cifar10/', rank=rank, train=False, download=False, transform=transform_c)

    elif args.dataset == 'tiny':
        print('Tiny')
        # dataset = TINYIMAGENET(root='../data/TINYIMAGENET/train/', train=True, transform = transform_train_t)
        # testset = TINYIMAGENET(root='../data/TINYIMAGENET/val/images', train=False,  transform=transform_val_t)
        dataset = TINYIMAGENET(root='../../data/TINYIMAGENET/train/', train=True, transform = transform_train_t)
        testset = TINYIMAGENET(root='../../data/TINYIMAGENET/val/images', train=False,  transform=transform_val_t)

    else:
        raise  NotImplementedError('Unsupported Datasets!')

    train_set = torch.utils.data.DataLoader(dataset,
                                         batch_size=args.batch_size,
                                         shuffle=True,   #For more stable results, shuffle can turn to False.
                                         num_workers=4)

    test_set = torch.utils.data.DataLoader(testset,
                                         batch_size=args.test_batch_size,
                                         # shuffle=True,   #For more stable results, shuffle can turn to False.
                                         num_workers=4)

    return train_set, test_set

""" Distributed Synchronous """
def run(rank, size, args):

    rank = dist.get_rank()
    train_set, test_set  =  partition_dataset(args)

    device = get_default_device(rank)

    if args.dataset == 'fmnist':  
        model = FashionMNISTModel() 
    elif args.dataset == 'cifar10':  
        model = CIFARModel()     
    elif args.dataset == 'tiny':  
        model = resnet18()
    else:
        print("ERRORRRR")
        return

    if args.init:
        print("Wegith Init")
        fname = 'model/' + args.dataset + '.pth'
        torch.save(model.state_dict(), fname)
    else:
        print("Wegith Loaded")
        fname = 'model/' + args.dataset + '.pth'
        model.load_state_dict(torch.load(fname))

    model = model.to(device)

    if args.method == 'localsgds':
        #when glr = 1 fedsgda ==> localsgds
        FSGDA(train_set, test_set, model, args, device) 
    elif args.method == 'fedsgda':
        FSGDA(train_set, test_set, model, args, device) 
    else:
        print("ERRORRRR")

    print("FL done")


def init_process(rank, size, args, fn, backend='gloo'):
# def init_process(rank, size, args, fn, backend='nccl'):
    """ Initialize the distributed environment. """
    os.environ['MASTER_ADDR'] = '127.0.0.1'
    os.environ['MASTER_PORT'] = str(args.port)
    dist.init_process_group(backend, rank=rank, world_size=size)
    fn(rank, size, args)


if __name__ == "__main__":
    # Training settings
    parser = argparse.ArgumentParser(description='PyTorch MNIST Example')
    parser.add_argument('--batch-size', type=int, default=100, metavar='N',
                        help='input batch size for training (default: 1000)')
    parser.add_argument('--test-batch-size', type=int, default=500, metavar='N',
                        help='input batch size for testing (default: 5000)')
    parser.add_argument('--epochs', type=int, default=1, metavar='N',
                        help='number of epochs to train (default: 10)')
    parser.add_argument('--worker-size', type=int, default=2, metavar='N',
                        help='szie of worker (default: 4)')

    parser.add_argument('--lr', type=float, default=0.001, metavar='LR',
                        help='learning rate (default: 0.1)')
    parser.add_argument('--lr2', type=float, default=0.0001, metavar='LR',
                        help='learning rate (default: 0.1)')
    parser.add_argument('--glr', type=float, default=1, metavar='LR',
                        help='learning rate (default: 0.1)')
    parser.add_argument('--alpha', type=float, default=0.1, metavar='alpha',
                        help='momentum rate alpha')
    parser.add_argument('--beta', type=float, default=0.1, metavar='alpha',
                        help='momentum rate beta')
    parser.add_argument('--lmd', type=float, default=0.001, metavar='alpha',
                        help='momentum rate beta')
    parser.add_argument('--rho', type=float, default=0.1, metavar='alpha',
                        help='momentum rate rho')

    parser.add_argument('--otLoop', type=int, default=1, metavar='S',
                        help='inter loop number')
    parser.add_argument('--inLoop', type=int, default=10, metavar='S',
                        help='inter loop number')
    parser.add_argument('--init', action='store_true', default=False,
                        help='For Saving the current Model')
    parser.add_argument('--dataset', type=str, default='mnist',
                        help='Dataset for trainig')
    parser.add_argument('--method', type=str, default='fedavg',
                        help='Dataset for trainig')
    parser.add_argument('--seed', type=int, default=0, metavar='S',
                        help='random seed (default: 1234)')
    parser.add_argument('--port', type=int, default=29505, metavar='S',
                        help='random seed (default: 29505)')
    args = parser.parse_args()
    print(args)

    size = args.worker_size 
    processes = []
    for rank in range(size):
        p = Process(target=init_process, args=(rank, size, args, run))
        p.start()
        processes.append(p)

    for p in processes:
        p.join()