import torch
import torch.nn as nn
# import torch.nn.functional as F
import sys

# re-direct stdout to a txt file
orig_stdout = sys.stdout
f = open('q1_out_res.txt', 'w')
sys.stdout = f

# initialize the data and hypre-params
t_data = torch.tensor([[1., 1.],[1., 0.], [0., 1.], [0., 0.]])
t_label = torch.tensor([0., 1., 1., 0.]).reshape([4,1])
learning_rate = 1e-1
n_epoches = 50

# model class
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.fc1 = nn.Linear(2, 3)
        self.fc2 = nn.Linear(3, 1)

    def forward(self, x):
        x = torch.tanh(self.fc1(x))
        x = self.fc2(x)
        return x

# train and evaluate
net = Net()
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(net.parameters(), lr=learning_rate)

for epoch in range(n_epoches):
    for i in range(len(t_label)):
        data = t_data[i]
        label = t_label[i]

        optimizer.zero_grad()

        # forward + backward + optimize
        outputs = net(data)
        train_loss = criterion(outputs, label)
        train_loss.backward()
        optimizer.step()

        # print train statistics
        print('train loss: {0:3.4f}'.format(train_loss))

    # print test statistics for the epoch
    outputs = net(t_data)

    test_loss = criterion(outputs, t_label)
    print('#### Epoch number {0} test loss {1:3.3f} ####'.format(epoch, test_loss))

sys.stdout = orig_stdout
f.close()

