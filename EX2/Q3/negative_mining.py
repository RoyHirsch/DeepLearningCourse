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

FC_STATE_DICT_PATH = '/Users/royhirsch/Documents/GitHub/DeepLearningCourse/EX2/Q1/model_params_test_loss_0.0813.pt'
FDDB_IMAGE_ORDER = '/Users/royhirsch/Documents/Study/Current/DeepLearning/Ex2/EX2_data/fddb/FDDB-folds/FDDB-fold-01.txt'
FDDB_IMAGES_ROOT = '/Users/royhirsch/Documents/Study/Current/DeepLearning/Ex2/EX2_data/fddb/images'
SCALES_LIST      = [10, 12]

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
patches = []
for im_name in images_iterator:
	patch_per_img = []
	print('Process image {}'.format(im_name))

	patch_per_scale = []
	for scale in SCALES_LIST:
		scaled_im = transforms.Resize(int(im.size[1]/scale))(im)
		h_input, w_input = scaled_im.size
		im_tensor = transforms.ToTensor()(scaled_im).view([1, im.layers, w_input, h_input])

		# Evaluate the FCN
		sigmoid = nn.Sigmoid()
		output = sigmoid(fcn_net(im_tensor))
		scores = np.squeeze(output.detach().numpy())[1, :, :]

		# Get the positive samples
		pos_rects = scores_to_boxes(scores)
		orig_pos_rects = pos_rects[:, :-1] * scale

		patch_per_img.append(orig_pos_rects.astype(np.int))

	# print('Got {} patches'.format(len(np.array(patch_per_img))))
	patches.append([im_name, np.concatenate(patch_per_img)])

pickle.dump(patches, open(os.path.join(''), 'wb'))

# Return patches - list with a tupel per sampel,
# each tupel: (image_name, pos_rect)
# where pos_rects is np.array shape: num_rects X 4_cordinates (x_top_left, y_top_left, h, w)