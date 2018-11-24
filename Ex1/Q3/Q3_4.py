import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import pandas as pd
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.sampler import SubsetRandomSampler
import matplotlib.pyplot as plt
import time

valid_samples = 2140
test_par = 0.2
n_epoches = 100  # changed from 400
batch_size = 32  # changed from 128

# csv_path = '/Users/royhirsch/Documents/Study/Current/DeepLearning/Ex1/EX1/Q3/training.csv'
csv_path = 'training.csv'

class FaceDataLoader(Dataset):

    def __init__(self, csv_file, transform=None):
        """
        DataLoader class for handling and pre-processing of the data
        Args:
            csv_file (string): Path to the csv file with annotations.
        """
        self.raw_data = pd.read_csv(csv_file).dropna()
        self.transform = transform
        self.col_names = ['left_eye_center_x', 'left_eye_center_y', 'right_eye_center_x',
                          'right_eye_center_y', 'left_eye_inner_corner_x',
                          'left_eye_inner_corner_y', 'left_eye_outer_corner_x',
                          'left_eye_outer_corner_y', 'right_eye_inner_corner_x',
                          'right_eye_inner_corner_y', 'right_eye_outer_corner_x',
                          'right_eye_outer_corner_y', 'left_eyebrow_inner_end_x',
                          'left_eyebrow_inner_end_y', 'left_eyebrow_outer_end_x',
                          'left_eyebrow_outer_end_y', 'right_eyebrow_inner_end_x',
                          'right_eyebrow_inner_end_y', 'right_eyebrow_outer_end_x',
                          'right_eyebrow_outer_end_y', 'nose_tip_x', 'nose_tip_y',
                          'mouth_left_corner_x', 'mouth_left_corner_y', 'mouth_right_corner_x',
                          'mouth_right_corner_y', 'mouth_center_top_lip_x',
                          'mouth_center_top_lip_y', 'mouth_center_bottom_lip_x',
                          'mouth_center_bottom_lip_y', 'Image']
        self.data = self.raw_data[self.col_names[-1]].iloc[:valid_samples].values
        self.labels = self.raw_data[self.col_names[:-1]].iloc[:valid_samples].values

    def __len__(self):
        return len(self.data)

    # Getter, normalizes the data and the labels
    def __getitem__(self, idx):
        sample = self.data[idx]
        sample_labels = self.labels[idx]
        norm_sample = np.fromstring(sample, sep=' ') / 255.
        norm_sample = norm_sample.reshape([1, 96, 96])
        norm_labels = (sample_labels - 48) / 48

        if self.transform:
            norm_sample = self.transform(norm_sample)

        return torch.tensor(norm_sample).float(), torch.tensor(norm_labels).float()

class CnnNet(nn.Module):
    def __init__(self):
        super(CnnNet, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, 3)
        self.pool1 = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(32, 64, 2)
        self.pool2 = nn.MaxPool2d(2, 2)
        self.conv3 = nn.Conv2d(64, 128, 2)
        self.pool3 = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(15488, 500)
        self.fc2 = nn.Linear(500, 500)
        self.fc3 = nn.Linear(500, 30)

    def forward(self, x):
        x = self.conv1(x)
        x = F.relu(self.pool1(x))
        x = self.conv2(x)
        x = F.relu(self.pool2(x))
        x = self.conv3(x)
        x = F.relu(self.pool3(x))
        x = x.view(-1, 15488)

        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

net = CnnNet()
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(net.parameters(), lr=0.001)
# optimizer = torch.optim.SGD(net.parameters(), lr=0.01, momentum=0.9, nesterov=True)

# Creating data indices for training and test splits:
test_size = int(test_par * valid_samples)
train_size = valid_samples - test_size

train_ind, test_ind = torch.utils.data.random_split(range(valid_samples), [train_size, test_size])
train_sampler = SubsetRandomSampler(train_ind)
test_sampler = SubsetRandomSampler(test_ind)

# Creating the train and test data loaders
face_dataset = FaceDataLoader(csv_file=csv_path)

# Use torch.utils.data.DataLoader iterator class as a wrapper for convenient batching and shuffling
train_loader = DataLoader(face_dataset, batch_size=batch_size, num_workers=0, sampler=train_sampler)
test_loader = DataLoader(face_dataset, batch_size=test_size, num_workers=0, sampler=test_sampler)


# Main optimization loop:
train_loss_arr = []
test_loss_arr = []
for epoch in range(n_epoches):
    start = time.time()
    for i, data in enumerate(train_loader, 0):
        inputs, labels = data

        optimizer.zero_grad()

        outputs = net(inputs)
        train_loss = criterion(outputs, labels)
        train_loss.backward()
        optimizer.step()

    # Test the model
    for data in test_loader:
        inputs, labels = data
        
    outputs = net(inputs)
    test_loss = (criterion(outputs, labels))

    end = time.time()
    if epoch % 10 == 0:
        print('\n==> Epoch num {} : time {} s'.format(epoch, round(end-start, 1)))
        print('train loss : {0:3.4f}'.format(train_loss))
        print('test loss : {0:3.4f}'.format(test_loss))

    # Document the results
    train_loss_arr.append(train_loss)
    test_loss_arr.append(test_loss)
    
# Final loss report
epoch_loss_arr = []
for data in train_loader:
    inputs, labels = data
    outputs = net(inputs)
    train_loss = (criterion(outputs, labels))
    epoch_loss_arr.append(train_loss.item())

train_loss = np.mean(epoch_loss_arr)
print('\n== Final epoch: ==\ntrain loss : {0:3.4f}\ntest loss : {1:3.4f}'.format(train_loss, test_loss))

# Plot the loss per epoch
plt.figure()
plt.plot(train_loss_arr, '-', color='b')
plt.plot(test_loss_arr, '--', color='r')
plt.legend(['train_loss', 'test_loss'])
plt.title('Loss per epoch, CNN network')
plt.xlabel('# Epoch')
plt.ylabel('Loss')
plt.show()
