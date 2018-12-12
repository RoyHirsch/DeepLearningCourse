import torch
import torchfile
from torch.utils.data import Dataset, DataLoader
import os
import numpy as np
from PIL import Image
from torchvision import transforms
from torch.utils.data.sampler import SubsetRandomSampler
import torch.nn as nn
#import torch.nn.functional as F
from itertools import cycle
import matplotlib.pyplot as plt
import pandas as pd
#import re
import pickle

''' ###################################### PARAMETERS ###################################### '''
# aflw_path = 'C:/Users/nimro/Desktop/ex2_local/EX2_data/EX2_data/aflw'
#aflw_path = '/Users/royhirsch/Documents/Study/Current/DeepLearning/Ex2/EX2_data/aflw'
aflw_path = 'C:/Users/dorim/Desktop/DOR/TAU uni/Msc/DL/EX2/EX2_data/aflw'
filename = 'aflw_24.t7'
pascal_path = 'C:/Users/dorim/Desktop/DOR/TAU uni/Msc/DL/EX2/EX2_data/VOCdevkit/VOC2007/JPEGImages'
negative_rects_pkl = 'C:/Users/dorim/Documents/GitHub/DeepLearningCourse/EX2/Q3/negative_rects.pkl'
#pascal_path = '/Users/royhirsch/Documents/Study/Current/DeepLearning/Ex2/VOCdevkit/VOC2007'
test_par = 0.1
dropout_par = 0.25
batch_size_pos = 96
batch_size_neg = 96
LR = 0.001
n_epoches = 100


''' ###################################### CLASSES ###################################### '''


class Aflw_loader(Dataset):
	'loads the 24*24 images as a num_sampels * dimension numpy and prepars'
	'a labels vector of size num_sampels  and samples a tensor of each'

	def __init__(self, path, filename):
		self.rawdata = torchfile.load(os.path.join(path, filename), force_8bytes_long=True)
		self.rawdata = list(self.rawdata.values())
		"""train at home on a partial dataset"""
		#self.rawdata = self.rawdata[0:10000]
		self.labels = np.ones((np.shape(self.rawdata)[0], 1))

	def __len__(self):
		return len(self.rawdata)

	def __getitem__(self, idx):
		sample = self.rawdata[idx]
		sample_labels = self.labels[idx]
		if torch.cuda.is_available():
			return torch.Tensor.cuda(torch.tensor(sample).float()), torch.Tensor.cuda(torch.tensor(sample_labels).float())
		else:
			return torch.tensor(sample).float(), torch.tensor(sample_labels).float()


def get_rects(rects_path):
	columns_list = ['image_name', 'rect']
	data_frame = pd.DataFrame([], columns=columns_list)

	rects_data = pickle.load(open(str(rects_path), "rb"))

	for sample in rects_data:
		im_name = sample[0]
		rects = sample[1]
		temp = pd.DataFrame([[im_name, rect] for rect in rects], columns=columns_list)
		data_frame = data_frame.append(temp, ignore_index=True)
	return data_frame

class mining_loader(Dataset):
	def __init__(self, path_rects, images_root):
		self.path_rects = path_rects
		self.images_root = images_root
		self.rects = get_rects(self.path_rects)

		self.transform = transforms.Compose([transforms.Resize((24,24)),transforms.ToTensor()])
	def __len__(self):
		return len(self.rects.index)

	def __getitem__(self, idx):
		im_name, rect = self.rects.loc[idx]
		#avoid taking rects < 24*24
		w = max(24, rect[2] - rect[0]+ 1)
		h = max(24, rect[3] - rect[1] + 1)
		fullim_name = os.path.join(self.images_root, im_name)
		cropped_image = transforms.functional.crop(Image.open(fullim_name),rect[0],rect[1],h,w)
		#interpolate to 24*24 image tensor
		sample = self.transform(cropped_image)
		if torch.cuda.is_available():
			return torch.Tensor.cuda(sample.float()), torch.Tensor.cuda(torch.tensor(np.array([0])).float())
		else:
			return sample.float(), torch.tensor(np.array([0])).float()

# change to 24X24 net
class Net(nn.Module):
	def __init__(self):
		super(Net, self).__init__()
		# [5,5] kernel ,output chanel: 64
		self.conv = nn.Conv2d(3, 64, 5, stride = 1 , padding =2)
		# output shape is 64x24x24
		self.dropout1 = nn.Dropout2d(p=0.5, inplace=True)
		self.pool = nn.MaxPool2d((3, 3), stride=2,padding = 1)
		self.relu = nn.ReLU(inplace=True)
		self.dropout2 = nn.Dropout2d(p=0.5, inplace=False)
		# output shape is 64x12x12
		self.fc1 = nn.Linear(64*144, 128)
		self.fc2 = nn.Linear(128, 2)
		self.sigmoid = nn.Sigmoid()

	def forward(self, x):
		if torch.cuda.is_available():
			dtype = torch.cuda.FloatTensor
			x = torch.autograd.Variable(x.type(dtype))
		x = self.conv(x)
		x = self.pool(self.dropout1(x))
		x = self.dropout2(self.relu(x))
		x = x.view(-1, 64*12*12)
		x = self.relu(self.fc1(x))
		x = self.sigmoid(self.fc2(x))
		return x

''' ###################################### FUNCTIONS ###################################### '''

def create_sampler_for_train_n_test(size_dataset, test_par):
	'''
	Creating data indices for training and test splits:
	:param size_dataset: (int) - num of samples in the dataset
	:param test_par: (float) - parcentage of data to extract for test
	:return: train_sampler, test_sampler
	'''
	ind = list(range(size_dataset))
	test_size = int(test_par * size_dataset)

	test_ind = np.random.choice(ind[0:int(2*test_par * size_dataset)], size=test_size, replace=False)
	train_ind = list(set(ind) - set(test_ind))

	train_sampler = SubsetRandomSampler(train_ind)
	test_sampler = SubsetRandomSampler(test_ind)
	return train_sampler, test_sampler,test_size

def permutate_input_n_labels(pos_inputs, pos_labels, neg_inputs, neg_labels):
	'''
    Gets positive and negative data and label and shuffle them,
    :param pos_inputs: (torch.tensor)
    :param pos_labels: (torch.tensor)
    :param neg_inputs: (torch.tensor)
    :param neg_labels: (torch.tensor)
    :return: shuffled data and labels
    '''
	merged_inputs = torch.cat((pos_inputs, neg_inputs))
	merged_labels = torch.cat((pos_labels, neg_labels))
	inds = list(range(len(merged_labels)))
	np.random.shuffle(inds)
	if torch.cuda.is_available():
		return torch.Tensor.cuda(merged_inputs[inds, :, :, :]).float(), torch.Tensor.cuda(merged_labels[inds, :]).flatten().long()
	else:
		return torch.tensor(merged_inputs[inds, :, :, :]).float(), torch.tensor(merged_labels[inds, :]).flatten().long()


''' ###################################### MAIN ###################################### '''

# Create the date loaders
positive_aflw_24_net = Aflw_loader(path=aflw_path, filename=filename)
negative_pascal_24_net = mining_loader(path_rects = negative_rects_pkl, images_root = pascal_path )

pos_train_sampler, pos_test_sampler,pos_test_size = create_sampler_for_train_n_test(len(positive_aflw_24_net.rawdata), test_par)
neg_train_sampler, neg_test_sampler, neg_test_size = create_sampler_for_train_n_test(len(negative_pascal_24_net.rects.iloc[:]), test_par)


pos_train_loader = DataLoader(positive_aflw_24_net, batch_size=batch_size_pos, sampler=pos_train_sampler)
neg_train_loader = DataLoader(negative_pascal_24_net, batch_size=batch_size_neg, sampler=neg_train_sampler)

#perform test in batches to avoid out of memory error
pos_test_loader = DataLoader(positive_aflw_24_net, batch_size= batch_size_pos, sampler=pos_test_sampler)
neg_test_loader = DataLoader(negative_pascal_24_net, batch_size= batch_size_neg, sampler=neg_test_sampler)

# Create net
net = Net()
criterion = nn.CrossEntropyLoss()
if torch.cuda.is_available():
	net = net.cuda()
	criterion = nn.CrossEntropyLoss().cuda()
optimizer = torch.optim.Adam(net.parameters(), lr=LR)

train_loss_total = []
test_loss_total = []
train_accuracy_total = []
test_accuracy_total = []

for epoch in range(n_epoches):
	train_loss_arr = []
	train_accuracy_arr = []
	# Because neg_train_loader iterator is smaller - use cycle iterator
	for i, data in enumerate(zip(pos_train_loader, cycle(neg_train_loader)), 0):
		(pos_inputs, pos_labels), (neg_inputs, neg_labels) = data
		inputs, labels = permutate_input_n_labels(pos_inputs, pos_labels, neg_inputs, neg_labels)
		optimizer.zero_grad()

		net.train()
		outputs = net(inputs)
		preds = torch.argmax(outputs, dim=1)
		accuracy_per_minibatch = (preds == labels).sum().item() / inputs.size()[0]
		if torch.cuda.is_available():
			dtype = torch.cuda.LongTensor
			labels = torch.autograd.Variable(labels.type(dtype))
		train_loss_tmp = criterion(outputs, labels)
		train_loss_tmp.backward()
		optimizer.step()
		train_loss_arr.append(train_loss_tmp.item())
		train_accuracy_arr.append(accuracy_per_minibatch)

	train_loss_epoch = np.mean(train_loss_arr)
	train_accuracy_epoch = np.mean(train_accuracy_arr)
	train_loss_total.append(train_loss_epoch)
	train_accuracy_total.append(train_accuracy_epoch)
	# Test the model
	test_loss_arr = []
	test_acc_arr = []
	for data in zip(pos_test_loader, cycle(neg_test_loader)):
		(pos_inputs, pos_labels), (neg_inputs, neg_labels) = data
		inputs, labels = permutate_input_n_labels(pos_inputs, pos_labels, neg_inputs, neg_labels)

		net.eval()
		outputs = net(inputs)
		preds = torch.argmax(outputs, dim=1)
		accuracy_per_minibatch = (preds == labels).sum().item() / inputs.size()[0]
		if torch.cuda.is_available():
			dtype = torch.cuda.LongTensor
			labels = torch.autograd.Variable(labels.type(dtype))
		test_loss_t = criterion(outputs, labels)
		test_loss_arr.append(test_loss_t.item())
		test_acc_arr.append(accuracy_per_minibatch)
	test_loss_epoch = np.mean(test_loss_arr)
	test_accuracy_epoch = np.mean(test_acc_arr)
	test_accuracy_total.append(test_accuracy_epoch)
	test_loss_total.append(test_loss_epoch)
	print('\n==> Epoch num {} :'.format(epoch))
	print('train loss : {0:3.4f} train accuracy: {1:3.4f} '.format(train_loss_epoch,train_accuracy_epoch))
	print('test loss : {0:3.4f} test accuracy: {1:3.4f} '.format(test_loss_epoch,test_accuracy_epoch))

# Save state_dict
torch.save(net.state_dict(), 'model_params_test_loss_{}.pt'.format(round(test_loss_epoch,4)))
# Plot the loss per epoch
plt.figure()
plt.plot(train_loss_total, '-', color='b')
plt.plot(test_loss_total, '--', color='r')
plt.legend(['train_loss', 'test_loss'])
plt.title('Loss per epoch graph')
plt.xlabel('# Epoch')
plt.ylabel('Loss')
plt.show()

plt.figure()
plt.plot(train_accuracy_total, '-', color='b')
plt.plot(test_accuracy_total, '--', color='r')
plt.legend(['train accuracy', 'test accuracy'])
plt.title('accuracy per epoch graph')
plt.xlabel('# Epoch')
plt.ylabel('accuracy')
plt.show()