# python main.py --method fscg  --epochs 1000 --inLoop 100 --inbs 10 --lr 0.0001
import os
import torch
import torch.distributed as dist
from torch.multiprocessing import Process
from torchvision import datasets, transforms
import torch.optim as optim
from dataset import *
from FCSGM import *

import random
import numpy as np
import argparse
import time


def get_default_device(idx):
    # print(idx)
    if torch.cuda.is_available():
        print("GPU available")
        return torch.device('cuda:' + str(idx))
    else:
        print("GPU NOT available")
        return torch.device('cpu')

""" Distributed Synchronous """
def run(rank, size, args):
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)

    rank = dist.get_rank()

    # div_pat = dist.get_world_size() / 2
    device = get_default_device(0)

    model  = LogModel(args.dim)
    tmodel = LogModel(args.dim)
    tmodel.evaluation()
    # for name, param in tmodel.named_parameters():
    #     print(name, param)

    if args.init and rank == 0:
        print("Wegith Init")
        fname = args.logdir + '/model.pth'
        torch.save(tmodel.state_dict(), fname)

        genTestData(tmodel, args.testbs)

    print("Wegith Loaded")
    fname = args.logdir + '/model.pth'
    tmodel.load_state_dict(torch.load(fname))

    model  = model.to(device)
    # tmodel = tmodel.to(device)
    datasets = getData(tmodel, args.inbs, args.dim, args.std)

    if args.method == 'fcsg':
        # alpha = 1 : fscgm > fscg
        FCSGM(model, datasets, args, device)
    elif args.method == 'fcsgm':
        FCSGM(model, datasets, args, device)
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
    parser.add_argument('--inbs', type=int, default=100, metavar='N',
                        help='input batch size for training (default: 1000)')
    parser.add_argument('--testbs', type=int, default=5000, metavar='N',
                        help='input batch size for testing (default: 5000)')
    parser.add_argument('--epochs', type=int, default=1, metavar='N',
                        help='number of epochs to train (default: 10)')
    parser.add_argument('--inLoop', type=int, default=100, metavar='S',
                        help='inter loop number')

    parser.add_argument('--worker-size', type=int, default=16, metavar='N',
                        help='szie of worker (default: 4)')

    parser.add_argument('--lr', type=float, default=0.001, metavar='LR',
                        help='learning rate (default: 0.1)')
    parser.add_argument('--std', type=float, default=1, metavar='standard varation',
                        help='std_2')
    parser.add_argument('--alpha', type=float, default=0.1, metavar='alpha',
                        help='momentum rate alpha')
    parser.add_argument('--gamma', type=float, default=10, metavar='alpha',
                        help='momentum rate alpha')
    parser.add_argument('--lmd', type=float, default=0.001, metavar='alpha',
                        help='momentum rate alpha')
    parser.add_argument('--dim', type=int, default=10, metavar='dimension',
                        help='dimension of data')


    parser.add_argument('--init', action='store_true', default=False,
                        help='For Saving the current Model')
    parser.add_argument('--logdir', type=str, default='model',
                        help='model dir')
    parser.add_argument('--method', type=str, default=20,
                        help='fcsg/fcsg-m')
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