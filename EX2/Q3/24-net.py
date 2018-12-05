import torch
import torchfile
from torch.utils.data import Dataset, DataLoader
import os
import numpy as np
from PIL import Image
from torchvision import transforms
from torch.utils.data.sampler import SubsetRandomSampler
import torch.nn as nn
import torch.nn.functional as F
from itertools import cycle
import matplotlib.pyplot as plt
import pandas as pd
import re
import pickle

''' ###################################### PARAMETERS ###################################### '''
# aflw_path = 'C:/Users/nimro/Desktop/ex2_local/EX2_data/EX2_data/aflw'
aflw_path = '/Users/royhirsch/Documents/Study/Current/DeepLearning/Ex2/EX2_data/aflw'
filename = 'aflw_12.t7'
# pascal_path = 'C:/Users/dorim/Desktop/DOR/TAU uni/Msc/DL/EX2/EX2_data/VOCdevkit/VOC2007'
pascal_path = '/Users/royhirsch/Documents/Study/Current/DeepLearning/Ex2/VOCdevkit/VOC2007'
test_par = 0.1
batch_size_pos = 32
batch_size_neg = 96
n_epoches = 201

''' ###################################### CLASSES ###################################### '''

class Aflw_loader(Dataset):
	'loads the 12*12 images as a num_sampels * dimension numpy and prepars'
	'a labels vector of size num_sampels  and samples a tensor of each'

	def __init__(self, path, filename):
		self.rawdata = torchfile.load(os.path.join(path, filename), force_8bytes_long=True)
		self.rawdata = self.rawdata.values()
		self.labels = np.ones((np.shape(self.rawdata)[0], 1))

	def __len__(self):
		return len(self.rawdata)

	def __getitem__(self, idx):
		sample = self.rawdata[idx]
		sample_labels = self.labels[idx]
		return torch.tensor(sample).float(), torch.tensor(sample_labels).float()

def load_pascal_to_numpy(path):
    'returns pascal images not containing a person as a list of numpy arrays of different sizes'
    'since the image size is not uniform, and a numpy array of the image labels = 0 , negative'
    images_path = os.path.join(path, "JPEGImages")
    images = [os.path.join(images_path, f) for f in os.listdir(images_path)]
    person_list_path = os.path.join(path, "ImageSets/Main/person_trainval.txt")
    person_table = np.fromfile(person_list_path, sep=' ').reshape(-1,2)
    'according to PASCAL VOC 2007 documentation there are three ground truth labels: -1: Negative, 1: Positive, 0:Difficult'
    no_person = []
    images_no_person = []
    for i in range(person_table.shape[0]):
        if person_table[i,1] != -1.0:
            no_person.append(int(person_table[i,0]))
    for image_name in images:
        head, tail = os.path.split(image_name)
        image_num = os.path.splitext(tail)[0]
        'add only the photo names with no person appearing in them'
        if int(image_num) in no_person:
            images_no_person.append(image_name)
    num_images = len(images_no_person)
    pascal_as_list = np.array([np.array(Image.open(fname)) for fname in images_no_person])
    labels = np.zeros((num_images, 1))
    return pascal_as_list, labels

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
		self.pascal_images, self.pascal_labels = load_pascal_to_numpy(self.images_root)

		self.rects = get_rects(self.path_rects)


	def __len__(self):
		return len(self.rawdata)

	def __getitem__(self, idx):
		im_name, rect = self.rects.loc[idx]
		# img = self.resized_images.loc[self.resized_images['image_name'] == name]['image'].values[0]
		# sample_labels = self.labels[idx]
		# return torch.tensor(sample).float(), torch.tensor(sample_labels).float()

# change to 24X24 net
class Net(nn.Module):
	def __init__(self):
		super(Net, self).__init__()
		# [3,3] kernel ,output chanel: 16
		self.conv = nn.Conv2d(3, 16, 3)
		self.pool = nn.MaxPool2d((3, 3), stride=2)
		self.fc1 = nn.Linear(256, 16)
		self.fc2 = nn.Linear(16, 2)

	def forward(self, x):
		x = self.conv(x)
		x = self.pool(F.relu(x))

		x = x.view(-1, 256)

		x = F.relu(self.fc1(x))
		x = self.fc2(x)
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

	test_ind = np.random.choice(ind, size=test_size, replace=False)
	train_ind = list(set(ind) - set(test_ind))

	train_sampler = SubsetRandomSampler(train_ind)
	test_sampler = SubsetRandomSampler(test_ind)
	return train_sampler, test_sampler


''' ###################################### MAIN ###################################### '''

# Create the date loaders
positive_aflw_12_net = Aflw_loader(path=aflw_path, filename=filename)

pos_train_sampler, pos_test_sampler = create_sampler_for_train_n_test(len(positive_aflw_12_net.rawdata), test_par)

pos_train_loader = DataLoader(positive_aflw_12_net, batch_size=batch_size_pos, sampler=pos_train_sampler)
# neg_train_loader = DataLoader(negative_pascal_12_net, batch_size=batch_size_neg, sampler=neg_train_sampler)

pos_test_loader = DataLoader(positive_aflw_12_net, batch_size=batch_size_pos, sampler=pos_test_sampler)
# neg_test_loader = DataLoader(negative_pascal_12_net, batch_size=batch_size_neg, sampler=neg_test_sampler)
