import os
import argparse
import numpy as np
import  torch
from    torch import nn
from    torch import optim
from    torch.nn import functional as F
from    torch.utils.data import TensorDataset, DataLoader
from    torch import optim
from    copy import deepcopy

from torch.multiprocessing import Process
import torch.distributed as dist
from model import *

class Meta(nn.Module):
    """
    Meta Learner
    """
    def __init__(self, args, config):
        """
        :param args:
        """
        super(Meta, self).__init__()

        self.net = Learner(config, args.imgc)
        self.net_ = None
        
        #Param related to inner opt
        self.update_lr = args.update_lr #SGD by default
        self.update_step = args.update_step
        self.update_step_test = args.update_step_test
        self.beta = args.beta #inner state momentum coefficient

        self.momentum_in = None
        self.momentum_ot = None

        self.k_spt = args.k_spt #number of support samples per task
        self.task_num = args.task_num #number of tasks per meta step
        
        #Params related to meta optimizer
        self.opt = args.opt
        self.meta_lr = args.meta_lr
        self.alpha = args.alpha

    def average(self):
        size = dist.get_world_size()
        if self.opt in ['scgd', 'scgdm']:
            # print("average 1")
            for i in range(len(self.momentum_in)):
                # itm = self.momentum_in[i].detach_()
                # dist.all_reduce(itm, op=dist.ReduceOp.SUM)
                dist.all_reduce(self.momentum_in[i], op=dist.ReduceOp.SUM)
                self.momentum_in[i].div_(size)
        if self.opt in ['sgdm', 'scgdm', 'acc']:
            # print("average 2")
            for i in range(len(self.momentum_ot)):
                dist.all_reduce(self.momentum_ot[i], op=dist.ReduceOp.SUM)
                self.momentum_ot[i].div_(size)


    def forward(self, x_spt, y_spt, x_qry, y_qry):
        """
        :param x_spt:   [b, setsz, c_, h, w]
        :param y_spt:   [b, setsz]
        :param x_qry:   [b, querysz, c_, h, w]
        :param y_qry:   [b, querysz]
        :return:
        """
        tmp_grad, tmp_state, loss_value = self.trainHelper(self.net, x_spt, y_spt, x_qry, y_qry)
        self.momentum_in = deepcopy([itm.detach() for itm in tmp_state])
        # self.momentum_ot = deepcopy([itm.detach() for itm in tmp_grad])

        if self.opt == "acc":
            if self.net_ == None:
                self.net_ = deepcopy(self.net)
            else:
                tmp_grad_, _, _ = self.trainHelper(self.net_, x_spt, y_spt, x_qry, y_qry)

        with torch.no_grad():
            if self.opt in ['sgd', 'scgd']:
                for i, param in enumerate(self.net.parameters()):
                    param.data.add_(tmp_grad[i], alpha=-self.meta_lr)

            elif self.opt in ['sgdm', 'scgdm']:
                if self.momentum_ot == None:
                    self.momentum_ot = deepcopy([itm.detach() for itm in tmp_grad])
                    for i, param in enumerate(self.net.parameters()):
                        param.data.add_(self.momentum_ot[i], alpha=-self.meta_lr)
                else: 
                    for i, param in enumerate(self.net.parameters()):
                        self.momentum_ot[i].copy_((1 - self.alpha) * self.momentum_ot[i] + self.alpha * tmp_grad[i].data)
                        param.data.add_(self.momentum_ot[i], alpha=-self.meta_lr)

            elif self.opt == 'acc':
                if self.momentum_ot == None:
                    self.momentum_ot = deepcopy([itm.detach() for itm in tmp_grad])
                    for i, param in enumerate(self.net.parameters()):
                        param.data.add_(self.momentum_ot[i], alpha=-self.meta_lr)
                else:
                    self.net_.load_state_dict(deepcopy(self.net.state_dict()))
                    for i, param in enumerate(self.net.parameters()):
                        self.momentum_ot[i].copy_(tmp_grad[i].data + (1 - self.alpha) * (self.momentum_ot[i] - tmp_grad_[i].data))
                        param.data.add_(self.momentum_ot[i], alpha=-self.meta_lr)
            else:
                Print("ERROR!!!!!!!!!!!")
        return loss_value

    def trainHelper(self, net, x_spt, y_spt, x_qry, y_qry):

        task_num, setsz, c_, h, w = x_spt.size(); querysz = x_qry.size(1)

        # losses_q = [0 for _ in range(2)]  # record first and last loss
        # corrects = [0 for _ in range(2)]
        losses_q = 0  # record first and last loss
        corrects = 0

        tmp_state = [torch.zeros_like(p) for p in net.parameters()] # self.
        tmp_grad  = [torch.zeros_like(p) for p in net.parameters()]

        for i in range(task_num):

            fast_weights = list(map(lambda p: p, net.parameters()))
            # fast_weights = deepcopy([i for i in list(net.parameters())])

            # inner train : dafulat 1 step
            for k in range(self.update_step):
                logits = net(x_spt[i], fast_weights, bn_training=True)
                loss = F.cross_entropy(logits, y_spt[i])

                grad = torch.autograd.grad(loss, fast_weights, create_graph=True)
                fast_weights = list(map(lambda p: p[1] - self.update_lr * p[0], zip(grad, fast_weights)))

            if self.opt in ['sgd', 'sgdm', 'acc'] or self.momentum_in is None:
                # print("opt 1")
                u_state = [u.detach().clone().requires_grad_() for u in fast_weights]
            else:
                # print("opt 2")
                u_state = list(map(lambda p: (1 - self.beta) * p[0] + self.beta * p[1].detach().clone(), \
                    zip(self.momentum_in, fast_weights)))
                u_state = [u.detach().clone().requires_grad_() for u in u_state]

            logits_q = net(x_qry[i], u_state, bn_training=True)
            loss_q = F.cross_entropy(logits_q, y_qry[i]) 
            # losses_q[1] += loss_q.detach().clone()

            losses_q += loss_q.detach().clone()
            with torch.no_grad():
                pred_q = F.softmax(logits_q, dim=1).argmax(dim=1)

                correct = torch.eq(pred_q, y_qry[i]).sum().item()
                # corrects[1] = corrects[1] + correct
                corrects = corrects + correct

            grad_q = torch.autograd.grad(loss_q, u_state)
            grad = torch.autograd.grad(fast_weights, net.parameters(), grad_outputs=grad_q)

            tmp_grad = [tmp_g + fast_g.detach().clone()/task_num for tmp_g, fast_g in zip(tmp_grad, grad)]
            tmp_state = [tmp_st + state_cur.detach().clone()/task_num for tmp_st, state_cur in zip(tmp_state, u_state)]

            net.zero_grad()

        return tmp_grad, tmp_state, torch.tensor([corrects / (querysz * task_num), losses_q.item() / task_num], dtype=torch.float64)

    def finetunning(self, x_spt, y_spt, x_qry, y_qry):
        """
        :param x_spt:   [setsz, c_, h, w]
        :param y_spt:   [setsz]
        :param x_qry:   [querysz, c_, h, w]
        :param y_qry:   [querysz]
        :return:
        """
        assert len(x_spt.shape) == 4

        querysz = x_qry.size(0)

        losses_q = [0 for _ in range(self.update_step_test + 1)] 
        corrects = [0 for _ in range(self.update_step_test + 1)]

        # in order to not ruin the state of running_mean/variance and bn_weight/bias
        # we finetunning on the copied model instead of self.net
        net = deepcopy(self.net)

        # 1. run the i-th task and compute loss for k=0
        logits = net(x_spt)
        loss = F.cross_entropy(logits, y_spt)
        grad = torch.autograd.grad(loss, net.parameters())
        fast_weights = list(map(lambda p: p[1] - self.update_lr * p[0], zip(grad, net.parameters())))

        # this is the loss and accuracy before first update
        with torch.no_grad():
            # [setsz, nway]
            logits_q = net(x_qry, net.parameters(), bn_training=True)
            loss_q = F.cross_entropy(logits_q, y_qry)
            losses_q[0] += loss_q
            # [setsz]
            pred_q = F.softmax(logits_q, dim=1).argmax(dim=1)
            correct = torch.eq(pred_q, y_qry).sum().item()
            corrects[0] = corrects[0] + correct

        # this is the loss and accuracy after the first update
        with torch.no_grad():
            # [setsz, nway]
            logits_q = net(x_qry, fast_weights, bn_training=True)
            loss_q = F.cross_entropy(logits_q, y_qry)
            losses_q[1] += loss_q
            # [setsz]
            pred_q = F.softmax(logits_q, dim=1).argmax(dim=1)
            # scalar
            correct = torch.eq(pred_q, y_qry).sum().item()
            corrects[1] = corrects[1] + correct

        for k in range(1, self.update_step_test):
            # 1. run the i-th task and compute loss for k=1~K-1
            logits = net(x_spt, fast_weights, bn_training=True)
            loss = F.cross_entropy(logits, y_spt)
            # 2. compute grad on theta_pi
            grad = torch.autograd.grad(loss, fast_weights)
            # 3. theta_pi = theta_pi - train_lr * grad
            fast_weights = list(map(lambda p: p[1] - self.update_lr * p[0], zip(grad, fast_weights)))

            logits_q = net(x_qry, fast_weights, bn_training=True)
            # loss_q will be overwritten and just keep the loss_q on last update step.
            loss_q = F.cross_entropy(logits_q, y_qry)
            losses_q[k + 1] += loss_q

            with torch.no_grad():
                pred_q = F.softmax(logits_q, dim=1).argmax(dim=1)
                correct = torch.eq(pred_q, y_qry).sum().item()  # convert to numpy
                corrects[k + 1] = corrects[k + 1] + correct

        del net

        accs = np.array(corrects) / querysz
        losses = np.array([l.data.cpu().numpy().item() for l in losses_q])

        return accs[-1], losses[-1]
