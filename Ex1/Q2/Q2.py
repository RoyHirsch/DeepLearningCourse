import torch
import pandas as pd
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
import torch.optim as optim
from torch.autograd import Variable
import numpy as np
from torchvision import transforms
from matplotlib import pyplot as plt

#dataset was edited with title to each column in the csv file
class IrisDataset(Dataset):
    def __init__(self, csv_file, transform=None):
        self.data = pd.read_csv(csv_file)
        d = {'Iris-setosa': 0, 'Iris-virginica': 1, 'Iris-versicolor': 2}
        self.data = self.data.replace(d)
        self.transform = transform

    def __getitem__(self, index):
        input_numpy = self.data.iloc[index,0:4]
        input = torch.FloatTensor(input_numpy)
        label_numpy = [self.data.iloc[index, 4]]
        label = torch.FloatTensor(label_numpy)
        return input, label

    def __len__(self):
        return len(self.data)

# define the activation function
def MyReQU(x):
        x[x < 0] = 0
        z = x*x
        return z
# define the net with the wanted architecture
class ReQUNet(nn.Module):
    def __init__(self):
        super(ReQUNet, self).__init__()
        n_in, n_h, n_out = 4, 64, 3
        self.fc1 = nn.Linear(n_in, n_h, True)
        self.fc2 = nn.Linear(n_h, n_out, True)

    def forward(self, x):
        h = MyReQU(self.fc1(x))
        pred = self.fc2(h)
        soft = nn.Softmax()
        pred_for_acc = soft(pred)
        return pred , pred_for_acc

batch_size = 30
epoch_num = 75

# change the path according to your file location, notice you need to add a header row (headline to each column)
path = "C:/Users/dorim/Desktop/DOR/TAU uni/Msc/DL\EX1/EX1/Q2/iris.data.csv"
train_dataset = IrisDataset(csv_file=path, transform=None)
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)


requ_net = ReQUNet()
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(requ_net.parameters(), lr=1e-3)

loss_list = []
acc_list = []
for epoch in range(epoch_num):
    running_loss = 0.0
    running_acc = 0.0
    for i, data in enumerate(train_loader, 0):
        inputs, labels = data
        inputs = inputs.float()
        labels = labels.long().view([batch_size,])
        optimizer.zero_grad()
        preds, preds_for_acc = requ_net(inputs)
        _, argmax = preds_for_acc.max(-1)
        correct = (labels.eq(argmax.long())).sum()
        running_acc += correct.item()/batch_size
        loss = criterion(preds, labels.long())
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
    normalized_acc = running_acc/(i+1)
    normalized_loss = running_loss/(i+1)
    loss_list.append(normalized_loss)
    acc_list.append(normalized_acc)
    print('epoch: {0:1d} train loss: {1:f} training accuracy: {2:3f}'.format(epoch, normalized_loss,normalized_acc))

print('classifier is trained (:')

#Plot the loss per epoch
plt.figure()
plt.plot(range(len(loss_list)), loss_list, '-')
plt.title('Loss per epoch')
plt.grid()
plt.xlabel('# Epoch')
plt.ylabel('Loss')
plt.show()

#Plot the accuracy per epoch
plt.figure()
plt.plot(range(len(acc_list)), acc_list, '-')
plt.title('Accuracy per epoch')
plt.grid()
plt.xlabel('# Epoch')
plt.ylabel('Accuracy')
plt.show()


