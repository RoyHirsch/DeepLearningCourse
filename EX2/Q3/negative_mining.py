import torch
import os
import numpy as np
from PIL import Image
from torchvision import transforms, datasets
import torch.nn as nn
import torch.nn.functional as F
import PIL
import matplotlib.pyplot as plt
import cv2 as cv2
import math
import pickle
import pandas as pd
from net_24 import load_pascal_to_numpy
import torch
''' ###################################### CLASSES ###################################### '''

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        # [3,3] kernel ,output chanel: 16
        self.conv = nn.Conv2d(3, 16, 3)
        self.pool = nn.MaxPool2d((3, 3), stride=2)
        self.conv2 = nn.Conv2d(16, 16, 4)
        self.conv3 = nn.Conv2d(16, 2, 1)

    def forward(self, x):
        x = self.conv(x)
        x = self.pool(F.relu(x))        # x.shape = (128, 16, 4, 4)

        x = F.relu(self.conv2(x))
        x = self.conv3(x)
        return x

def scores_to_boxes(score_per_patch):
	'''
	Converts output scors from the NN to rects.
	Convert each cordinate from: [x, y, h, w] to [x1, y1, x2, y2]
	:param score_per_patch: (np.array)
	:param thres_percentile: min value for score to be observed as positive, based on a percentile
	:return:
	'''
	h, w = np.shape(score_per_patch)

	# Create array that holds the two points represent rectangle - top left and bottom right
	# Assume the location (i, j) in the output map is the top left corner at the input image at: (2i, 2j)
	# np.where returns a tupal of indexes 0 dim ix the x (row) ind and dim 1 is y ind (col)
	# top_left_ind.shape = [2, num_of_points]

	# Assume (i, j) is top left corner
	top_left_ind = np.array(np.where(score_per_patch >= 0.5))
	top_left_ind = top_left_ind * 2
	bottom_right_ind = top_left_ind + np.full(top_left_ind.shape, 12)
	filter_scores = score_per_patch[np.where(score_per_patch >= 0.5)]
	boxes = np.column_stack((top_left_ind.T, bottom_right_ind.T, filter_scores))

	return boxes

FC_STATE_DICT_PATH = 'C:/Users/dorim/Documents/GitHub/DeepLearningCourse/EX2/Q1/model_params_test_loss_0.0813.pt'
negative_root = 'C:/Users/dorim/Desktop/DOR/TAU uni/Msc/DL/EX2/EX2_data/VOCdevkit/VOC2007'
negative_dataframe = load_pascal_to_numpy(negative_root)
SCALES_LIST      = [8, 10]

''' ###################################### MAIN ###################################### '''

# Load the parameters of the FC network
fc_state_dict = torch.load(FC_STATE_DICT_PATH)

# Convert state_dict of the original FC NN into FCN
fcn_state_dict = fc_state_dict.copy()
fcn_state_dict['conv2.weight'] = fcn_state_dict.pop('fc1.weight').view(16, 16, 4, 4)
fcn_state_dict['conv2.bias'] = fcn_state_dict.pop('fc1.bias')

fcn_state_dict['conv3.weight'] = fcn_state_dict.pop('fc2.weight').view(2, 16, 1, 1)
fcn_state_dict['conv3.bias'] = fcn_state_dict.pop('fc2.bias')

fcn_net = Net()
fcn_net.load_state_dict(fcn_state_dict)

# Read the images by their order
#negative_dataframe = negative_dataframe.iloc[:300,:]
patches = []
for row in negative_dataframe.iterrows():
	patch_per_img = []
	print('Process image {}'.format(row[1][0]))
	full_im_path = os.path.join(negative_root +'/JPEGImages', row[1][0])
	im = Image.open(full_im_path)
	h_org, w_org = im.size
	patch_per_scale = []
	for scale in SCALES_LIST:
		scaled_im = transforms.Resize(int(im.size[1]/scale))(im)
		h_input, w_input = scaled_im.size

		# Convert gray scale input into 3 channel input
		if im.layers == 1:
			scaled_im = np.dstack((scaled_im, scaled_im, scaled_im))
		if row[1][0] == '001557.jpg':
			print('wait')
		im_tensor = transforms.ToTensor()(scaled_im).view([1, 3, w_input, h_input])
		# Evaluate the FCN

		sigmoid = nn.Sigmoid()
		output = sigmoid(fcn_net(im_tensor))
		scores = np.squeeze(output.detach().numpy())[1, :, :]


		# Get the positive samples
		pos_rects = scores_to_boxes(scores)
		orig_pos_rects = pos_rects[:, :-1] * scale

		patch_per_img.append(orig_pos_rects.astype(np.int))

	# print('Got {} patches'.format(len(np.array(patch_per_img))))
	patches.append([row[1][0], np.concatenate(patch_per_img)])

pickle.dump(patches, open(os.path.join(''), 'wb'))

# Return patches - list with a tupel per sampel,
# each tupel: (image_name, pos_rect)
# where pos_rects is np.array shape: num_rects X 4_cordinates (x_top_left, y_top_left, h, w)
