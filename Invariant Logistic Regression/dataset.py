import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable, Function
from functools import reduce
from operator import mul


class getData:

    def __init__(self, tmodel, inbs=1, dim=10, std=1):
        self.tmodel = tmodel
        self.inbs = inbs
        self.dim = dim
        self.std = std

    def getdata(self, bz=1):

        lst = []
        tgt = []
        for _ in range(bz):
            a = torch.normal(mean=torch.zeros(self.dim), std=torch.ones(self.dim))
            tgt.append(torch.unsqueeze(a, dim=0))

            for _ in range(self.inbs):
                # tmp = a
                tmp = torch.normal(mean=a, std=torch.ones(self.dim) * self.std)
                lst.append(torch.unsqueeze(tmp, dim=0))

        x = torch.cat(lst,0) 
        y = self.getTgt(torch.cat(tgt,0)).detach().repeat(1, self.inbs).view(-1, 1)
        # y = self.getTgt(x).detach()

        return x, y
    def getTgt(self, x):
        return torch.sign(self.tmodel(x))

    def getDif(self, model):

        lst = []
        for _ in range(self.inbs):
            tmp = torch.normal(mean=torch.zeros(self.dim), std=torch.ones(self.dim))
            lst.append(torch.unsqueeze(tmp, dim=0))
        x = torch.cat(lst,0)
        y = self.getTgt(x)


        loss_1 = -1 * np.mean(np.log(torch.sigmoid(model(data) * target) ))
        loss_2 = -1 * np.mean(np.log(torch.sigmoid(self.tmodel(data) * target)))

        return loss_1 - loss_2

# mu, sigma = 0, 0.1 # mean and standard deviation
# s = np.random.normal(mu, sigma, 1000)

def genTestData(model, bz, dim=10):
    lst = []
    for _ in range(bz):
        # tmp = a
        tmp = torch.normal(mean=torch.zeros(dim), std=torch.ones(dim))
        lst.append(torch.unsqueeze(tmp, dim=0))

    x = torch.cat(lst,0) 
    y = torch.sign(model(x)).detach()
    print("test data", x.shape, torch.sum(y))

    torch.save(x, str(dim) + 'testdata.pt')
    torch.save(y, str(dim) + 'testtgt.pt')

class LogModel(nn.Module):
    def __init__(self, input_dim = 10, tmodel=False):
        super().__init__()        
        self.fc_1 = nn.Linear(input_dim, 1)
        self._init_weights()
        # if tmodel:
        #     nn.init.constant_(self.fc_1.bias, 0)

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, (nn.Linear)):
                nn.init.xavier_normal_(m.weight)
                nn.init.constant_(m.bias, 1 / m.bias.numel())

    def evaluation(self):
        # self.fc_1.include_bias = False
        nn.init.constant_(self.fc_1.bias, 0)
        # for param in self.modules():
        for name, param in self.named_parameters():
            param.requires_grad = False

    def forward(self, x):
        return self.fc_1(x)

