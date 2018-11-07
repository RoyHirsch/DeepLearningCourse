import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import pandas as pd

from EX1.q2_utils import *


# load and pre-process the dataset
iris_dir = '/Users/royhirsch/Documents/Study/Current/DeepLearning/Ex1/EX1/Q2/iris.data.csv'
data = pd.read_csv(iris_dir)
label_dict = {'Iris-setosa': 0, 'Iris-versicolor': 1, 'Iris-virginica': 2}
data = data.replace(label_dict)

# extract data and label, convert to numpy arrays
label = data.iloc[:,-1].values
data = data.iloc[:,:-1].values

t_data = torch.tensor(data)
t_label = torch.tensor(label)
learning_rate = 1e-3
n_epoches = 50
batch_size = 32

def ReQU(x):
	w, h = x.size()
	x = x.view([1,-1])
	x[x < 0] = 0
	x = x * x
	return x.view([w,h])

# model class
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.fc1 = nn.Linear(4, 64)
        self.fc2 = nn.Linear(64, 128)

    def forward(self, x):
        # x = F.relu(self.fc1(x))
        x = ReQU(self.fc1(x))
        x = self.fc2(x)
        return x

# train and evaluate
net = Net()
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(net.parameters(), lr=learning_rate)

loss_arr = []
for epoch in range(n_epoches):
    t_data, t_label = randData(t_data, t_label)

    for i in range(len(t_label)//batch_size):
        data = t_data[i*batch_size:(i+1)*batch_size]
        label = t_label[i*batch_size:(i+1)*batch_size]
        optimizer.zero_grad()

        # forward + backward + optimize
        outputs = net(data)
        train_loss = criterion(outputs, label)
        train_loss.backward()
        optimizer.step()

        # print train statistics
        print('train loss: {0:3.4f}'.format(train_loss))
        loss_arr.append(train_loss)
