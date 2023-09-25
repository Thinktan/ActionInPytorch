import torch
from torch import nn
import sklearn
import pandas as pd
import numpy as np

print(sklearn.__version__)

class LinearModel(nn.Module):
    def __init__(self, ndim):
        super(LinearModel, self).__init__()
        self.ndim = ndim

        self.weight = nn.Parameter(torch.randn(ndim, 1))
        self.bias = nn.Parameter(torch.randn(1))

    def forward(self, x):
        # y = Wx + b
        return x.mm(self.weight) + self.bias




data_url = "http://lib.stat.cmu.edu/datasets/boston"
raw_df = pd.read_csv(data_url, sep="\s+", skiprows=22, header=None)
data = np.hstack([raw_df.values[::2, :], raw_df.values[1::2, :2]])
target = raw_df.values[1::2, 2]

print(data.shape)
print(target.shape)
print(data[0])
print(target[0])



lm = LinearModel(13)
criterion = nn.MSELoss()
optim = torch.optim.SGD(lm.parameters(), lr=1e-6)
data = torch.tensor(data, requires_grad=True, dtype=torch.float32)
target = torch.tensor(target, dtype=torch.float32)

for step in range(10000):
    predict = lm(data)
    loss = criterion(predict, target)
    if step and step % 1000 == 0:
        print("Loss: {:.3f}".format(loss.item()))
    optim.zero_grad()
    loss.backward()
    optim.step()


