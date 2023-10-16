import os
import argparse
import numpy as np
import  torch
from torch import nn
from torch import optim
from torch.nn import functional as F
from torch.utils.data import TensorDataset, DataLoader
from torch import optim
from copy import deepcopy

from torch.multiprocessing import Process
import torch.distributed as dist

from dataset import MiniImagenetNShot, OmniglotNShot
from model import *
from optimizer import * 
import random

def main(args):

    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)

    rank = dist.get_rank()
    world_size = dist.get_world_size()

    if args.data == 'mini':    
        maml = Meta(args, config_mini).cuda()
    elif args.data == 'om':    
        maml = Meta(args, config_om).cuda()
    else:
        print("ERROR!!!!")
        return
    
    train_lst = []; val_lst = []

    if args.data == 'mini':    
        db_train = MiniImagenetNShot(batchsz=args.task_num, n_way = args.n_way, k_shot=args.k_spt, k_query=args.k_qry, rank = rank, size = world_size)
    elif args.data == 'om':    
        db_train = OmniglotNShot(batchsz=args.task_num, n_way = args.n_way, k_shot=args.k_spt, k_query=args.k_qry, rank = rank, size = world_size)
    else:
        print("ERROR!!!!")
        return


    step = 0
    for batch in db_train.dataloader:

        x_spt, y_spt = batch['train']
        x_qry, y_qry = batch['test']
        x_spt, y_spt, x_qry, y_qry = x_spt.cuda(), y_spt.cuda(), x_qry.cuda(), y_qry.cuda()

        # train
        loss_value = maml(x_spt, y_spt, x_qry, y_qry)
        # train_acc.append(accs[-1]); train_loss.append(losses[-1])
            
        if step % args.local_epoch == 0:
            # Communication
            maml.net.average()
            maml.average()
            dist.all_reduce(loss_value, op=dist.ReduceOp.SUM)
            loss_value.div_(size)
            if rank == 0:
                train_lst.append([step, loss_value[0].item(), loss_value[1].item()])
                # print(step, loss_value[0].item(), loss_value[1].item())

        if rank == 0 and step % 50 == 0:
                
            accs = []; losses = []
            test_step = 0
            for test_batch in db_train.dataloader_val:
                x_spt, y_spt = test_batch['train']
                x_qry, y_qry = test_batch['test']
                x_spt, y_spt, x_qry, y_qry = x_spt.cuda(), y_spt.cuda(), x_qry.cuda(), y_qry.cuda()

                # split to single task each time
                for x_spt_one, y_spt_one, x_qry_one, y_qry_one in zip(x_spt, y_spt, x_qry, y_qry):
                    test_acc, test_loss = maml.finetunning(x_spt_one, y_spt_one, x_qry_one, y_qry_one)
                    accs.append(test_acc); losses.append(test_loss)
                
                test_step += args.task_num
                if test_step > 1:
                    break
            val_lst.append([step, accs[-1], losses[-1]])
            # print("val", step, accs[-1], losses[-1])
            # val_acc.append(accs[-1]); val_loss.append(losses[-1])
                

        step += 1
        if step > args.epoch:
            break

    if rank == 0:
        print("------ Train -----")
        for itm in train_lst:
            print(itm[0], itm[1], itm[2])
        print("------ Test -----")
        for itm in val_lst:
            print(itm[0], itm[1], itm[2])


def init_processes(rank, size, args, fn, backend='gloo'):
    """ Initialize the distributed environment. """
    os.environ['MASTER_ADDR'] = '127.0.0.1'
    os.environ['MASTER_PORT'] = args.port
    os.environ["CUDA_VISIBLE_DEVICES"] = str(rank % 2 + args.gpu)
    # print('hhhh')
    dist.init_process_group(backend, rank=rank, world_size=size)
    fn(args)


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--port', type=str, help='MASTER_PORT', default='24121')
    parser.add_argument('--gpu', type=int, help='MASTER_GPU', default=6)

    parser.add_argument('--size', type=int, help='world size', default=8)
    parser.add_argument('--data', type=str, help='data set', default='mini')

    parser.add_argument('--epoch', type=int, help='epoch number', default=500)
    parser.add_argument('--local_epoch', type=int, help='epoch number', default=10)
    parser.add_argument('--n_way', type=int, help='number classes', default=5)
    parser.add_argument('--k_spt', type=int, help='k shot for support set', default=1)
    parser.add_argument('--k_qry', type=int, help='number samples for query set', default=15)
    # parser.add_argument('--imgsz', type=int, help='imgsz', default=84)
    parser.add_argument('--imgc', type=int, help='imgc', default=3)
    parser.add_argument('--task_num', type=int, help='meta batch size, namely task num', default=5)
    parser.add_argument('--opt', type=str, help='sgd/momentum/adam', default='momentum')
    # OPt method: 
    # Local-SCGD - scgd; Local-SCGDM - scgdm
    # FCSG - sgd; FCSG-M - sgdm; Acc-FCSG-M - acc

    parser.add_argument('--update_lr', type=float, help='task-level inner update learning rate', default=0.4)
    parser.add_argument('--meta_lr', type=float, help='meta-level outer learning rate', default=0.1)
    parser.add_argument('--alpha', type=float, help='momentum coefficient for SCGD outer update', default=1)
    parser.add_argument('--beta', type=float, help='momentum coefficient for SCGD', default=1)

    parser.add_argument('--update_step', type=int, help='task-level inner update steps', default=1)
    parser.add_argument('--update_step_test', type=int, help='update steps for finetunning', default=3)

    parser.add_argument('--restore', dest='restore', action='store_true')
    parser.add_argument('--mult_state', dest='mult_state', action='store_true', help='maintain one inner state per task')
    parser.add_argument('--test_mode', dest='test_mode', action='store_true', help='generate test sinusoid curve')

    parser.add_argument('--seed', type=int, default=0, metavar='S',
                        help='random seed (default: 1234)')
    args = parser.parse_args()
    print(args)

    size = args.size
    processes = []

    for rank in range(size):
        p = Process(target=init_processes, args=(rank, size, args, main))
        p.start()
        processes.append(p)

    for p in processes:
        p.join()
