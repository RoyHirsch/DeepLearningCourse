import numpy as np
import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.optim as optim

class XOR_Net(nn.Module):
    def __init__(self):
            super(XOR_Net, self).__init__()
            D_in,H,Dout = 2,3,1
            self.fc1 = nn.Linear(D_in, H, True)
            self.fc2 = nn.Linear(H, Dout, True)
    def forward(self, x):
        m = nn.Tanh()
        x = m(self.fc1(x))
        x = self.fc2(x)

        return x

xor_net = XOR_Net()
X_train = list(map(lambda s: Variable(torch.Tensor([s])), [
[0, 0],
[0, 1],
[1, 0],
[1, 1]
]))
Y_train  = list(map(lambda s: Variable(torch.Tensor([s])), [
[0],
[1],
[1],
[0]
]))
num_epochs = 5000
criterion = nn.MSELoss()
optimizer = optim.Adam(xor_net.parameters(), lr=0.001)

for t in range(num_epochs):
    for sample,label in zip(X_train,Y_train):
        optimizer.zero_grad()
        y_pred = xor_net(sample)
        loss = criterion(y_pred,label)
        loss.backward()
        optimizer.step()
    if t%500 ==0:
        print("Epoch: %d , Loss: %f" %(t,loss))

#test the trained classefier for each row
for sample,label in zip(X_train,Y_train):
    y_pred = xor_net(sample)
    Sample_x1 = int(sample.data.numpy()[0][0])
    Sample_x2 = int(sample.data.numpy()[0][1])
    Target =  int(label.data.numpy()[0])
    Predicted = round(float(y_pred.data.numpy()[0]), 4)
    print("Sample: [{},{}] Target: [{}] Predicted: [{}]".format(Sample_x1,Sample_x2,Target,Predicted))
