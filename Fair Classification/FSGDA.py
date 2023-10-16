import os
import torch
import torch.distributed as dist
from torch.multiprocessing import Process
from torchvision import datasets, transforms
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

import numpy as np
import argparse
from sklearn.metrics import roc_auc_score
import copy

def FSGDA(train_set, test_set, model, args, device):
	if args.dataset == "tiny":
		print('fsgda, tiny, 0')
		class_ = 200
	else:
		print('fsgda, fmnist / cifar')
		class_ = 10		

	rank = dist.get_rank()
	size = dist.get_world_size()

	optimizer = optim.SGD(model.parameters(), lr=args.lr)

	wgt = torch.ones(class_, device = device)
	wgt = wgt / wgt.sum()
	model_ = copy.deepcopy(model)

	x_para = copy.deepcopy(list(map(lambda p: p, model.parameters())))
	y_para = copy.deepcopy(wgt)

	lss = 0

	Iteration = 0
	for epoch in range(args.epochs):
		for siter, (data, target) in enumerate(train_set):
			model.train()
			classes = np.unique(target.numpy())
			data   = data.to(device)
			target = target.to(device)

			output = model(data)

			loss = (-F.log_softmax(output, dim=1)).gather(1, target.view(-1, 1))
			loss_lst = []
			for cls_ in classes:
				loss_lst.append(loss[(target == cls_)].mean() * wgt[cls_])

			loss  = sum(loss_lst) # loss1 * t[0] + loss2 * t[1] + loss3 * t[2]
			loss.backward()
			lss += loss.detach().clone()

            # Update x
			optimizer.step()
			optimizer.zero_grad()
			# update y
			with torch.no_grad():
				output = model_(data)
				loss_w = (-F.log_softmax(output, dim=1)).gather(1, target.view(-1, 1))
				for cls_ in classes:
					wgt[cls_] = wgt[cls_] + args.lr2 * loss_w[(target == cls_)].mean() 

				wgt = F.relu(wgt)
				wgt = wgt / wgt.sum()

			#### testing  #######
			if Iteration % args.inLoop == 0:

				#### testing  #######
				model.eval()
				correct_cnt, ave_loss = 0, 0
				total_cnt = 0

				with torch.no_grad():
					for batch_idx, (data, target) in enumerate(test_set):
						data   = data.to(device)
						target = target.to(device)

						out = model(data)
						_, pred_label = torch.max(out.data, 1)
						total_cnt += data.data.size()[0]
						correct_cnt += (pred_label == target.data).sum()
				#########

				loss_value = torch.tensor([lss / args.inLoop, correct_cnt, total_cnt], dtype=torch.float64)
				dist.all_reduce(loss_value, op=dist.ReduceOp.SUM)
				loss_lst = loss_value.tolist()
				
				lss = 0
				if rank == 0:
					print(Iteration // args.inLoop, loss_lst[0] / size, loss_lst[1] / loss_lst[2] )	

			Iteration += 1

			# ### Communication ############
			if Iteration % args.inLoop == 0:
				for i, param in enumerate(model.parameters()):
					dist.all_reduce(param.data, op=dist.ReduceOp.SUM)
					param.data.div_(size)
					param.data.copy_(x_para[i] + args.glr * (param.data - x_para[i]))

				dist.all_reduce(wgt, op=dist.ReduceOp.SUM)
				wgt.div_(size)
				wgt.copy_(y_para + args.glr * (wgt - y_para))

				x_para = copy.deepcopy(list(map(lambda p: p, model.parameters())))
				y_para = copy.deepcopy(wgt)

				if Iteration % (args.otLoop * args.inLoop) == 0:
					model_.load_state_dict(copy.deepcopy(model.state_dict()))
