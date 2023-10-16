import os
import torch
import torch.distributed as dist
from torch.multiprocessing import Process
from torchvision import datasets, transforms
import torch.optim as optim
from dataset import *
import random
import torch.nn.functional as F
import numpy as np
import argparse
import time
import copy

def FCSGM(model, datasets, args, device):

	Iteration = 0
	rank = dist.get_rank()
	size = dist.get_world_size()
	m_t = None
	# [torch.zeros_like(param) for param in model.parameters()]

	if rank == 0:
		print("FCSGM", args.alpha)
		testData = torch.load(str(args.dim) + 'testdata.pt').to(device)
		testTgt  = torch.load(str(args.dim) + 'testtgt.pt').to(device)
		# print("Testdata", (testTgt == -1).sum())

	loss_t = 0

	for epoch in range(args.epochs):
		for siter in range(args.inLoop):
			model.train()

			# data, target = datasets.getdata(5)
			# print(np.shape(data), np.shape(target))
			# if rank == 0:
			# 	print(target)
			# return
			# data, target = data.to(device), target.to(device)

			# predScore = torch.sigmoid(model(data) * target)
			# loss = -1 * torch.mean(torch.log(predScore))
			if m_t == None:
				data, target = datasets.getdata(32)
			else:
				data, target = datasets.getdata()

			data, target = data.to(device), target.to(device)

			predScore = torch.sigmoid(model(data) * target)
			reg = 0
			for para in model.parameters():
				reg += torch.sum(args.gamma * para**2 / (1 + args.gamma * para**2))

			loss = -1 * torch.mean(torch.log(predScore)) + args.lmd * reg
			# loss = -1 * torch.mean(torch.log(predScore))
			loss.backward()

			if m_t == None:
				m_t = [param.grad.data for param in model.parameters()]
			# Update
			for i, param in enumerate(model.parameters()):

				m_t[i] = (1 - args.alpha) * m_t[i] + args.alpha * param.grad.data
				param.data.add_(m_t[i], alpha= - args.lr)

			model.zero_grad()
			loss_t += loss.item()

		# ### Communication ############
		printf = 5
		if (epoch + 1) % printf == 0:
			dist.all_reduce(torch.tensor([loss_t], dtype=torch.float64), op=dist.ReduceOp.SUM)
			if rank == 0:
				model.eval()
				with torch.no_grad():
					out = model(testData)
					pred_label = torch.sign(out)
					correct_cnt = (pred_label == testTgt).sum()
					acc = correct_cnt / testData.size()[0]

				print(epoch , loss_t / (printf * size * args.inLoop), acc.item())

			loss_t = 0

		for i, param in enumerate(model.parameters()):
			dist.all_reduce(param.data, op=dist.ReduceOp.SUM)
			dist.all_reduce(m_t[i], op=dist.ReduceOp.SUM)
			param.data.div_(size)
			m_t[i].div_(size)

