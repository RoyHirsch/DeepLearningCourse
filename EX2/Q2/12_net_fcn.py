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

''' ###################################### FUNCTIONS ###################################### '''

def square_to_elipse(square):
	'''

	:param square:
	:return: <major_axis_radius minor_axis_radius angle center_x center_y 1>.
	'''
	pass

def scores_to_boxes(score_per_patch, thres_percentile, h_input, w_input):
	'''
	Converts output scors from the NN to rects.
	Convert each cordinate from: [x, y, h, w] to [x1, y1, x2, y2]
	:param score_per_patch: (np.array)
	:param thres_percentile: min value for score to be observed as positive, based on a percentile
	:return:
	'''
	h, w = np.shape(score_per_patch)
	boxes = np.zeros([h * w, 5])

	# filer by score bigger then percentile
	m = np.percentile(score_per_patch.flatten(), thres_percentile)

	# Create array that holds the two points represent rectangle - top left and bottom right
	# Assume the location (i, j) in the output map is the top left corner at the input image at: (2i, 2j)
	# np.where returns a tupal of indexes 0 dim ix the x ind and dim 1 is y ind
	# top_left_ind.shape = [2, num_of_points]

	# Assume (i, j) is top left corner

	top_left_ind = np.array(np.where(score_per_patch >= m))
	top_left_ind = top_left_ind * 2
	bottom_right_ind = top_left_ind + np.full(top_left_ind.shape, 12)
	filter_scores = score_per_patch[np.where(score_per_patch >= m)]
	boxes = np.column_stack((top_left_ind.T, bottom_right_ind.T, filter_scores))

	# Assume (i, j) is bottom left corner
	
	# bottom_left_ind = np.array(np.where(score_per_patch >= m)) * 2
	# bottom_right_ind = np.copy(bottom_left_ind)
	# bottom_right_ind[0] = bottom_left_ind[0] + np.full(bottom_left_ind[0].shape ,12)
	#
	# top_left_ind = np.copy(bottom_left_ind)
	# top_left_ind[1] = bottom_left_ind[1] + np.full(bottom_left_ind[1].shape ,-12)
	# filter_scores = score_per_patch[np.where(score_per_patch >= m)]
	# boxes = np.column_stack((top_left_ind.T, bottom_right_ind.T, filter_scores))

	return boxes

def non_maxima_supration(boxes, thres=0.5):
	'''
	The output scores image of the backbone NN
	:param score_per_patch: (np.array)
	:param thres_percentile: min value for score to be observed as positive, based on a percentile
	:param thres: threshold for IOU of overlaping rects
	:return:
	'''
	# initialize the list of picked indexes
	pick = []

	# grab the coordinates of the bounding boxes
	x1 = boxes[:, 0]
	y1 = boxes[:, 1]
	x2 = boxes[:, 2]
	y2 = boxes[:, 3]
	scores = boxes[:, 4]

	# compute the area of the bounding boxes and sort the bounding
	# boxes by the bottom-right y-coordinate of the bounding box
	area = (x2 - x1 + 1) * (y2 - y1 + 1)
	idxs = np.argsort(y2)
	# keep looping while some indexes still remain in the indexes
	# list
	while len(idxs) > 0:
		# grab the last index in the indexes list, add the index
		# value to the list of picked indexes, then initialize
		# the suppression list (i.e. indexes that will be deleted)
		# using the last index
		last = len(idxs) - 1
		i = idxs[last]
		pick.append(i)
		suppress = [last]
		# loop over all indexes in the indexes list
		for pos in xrange(0, last):
			# grab the current index
			j = idxs[pos]

			# find the largest (x, y) coordinates for the start of
			# the bounding box and the smallest (x, y) coordinates
			# for the end of the bounding box
			xx1 = max(x1[i], x1[j])
			yy1 = max(y1[i], y1[j])
			xx2 = min(x2[i], x2[j])
			yy2 = min(y2[i], y2[j])

			# compute the width and height of the bounding box
			w = max(0, xx2 - xx1 + 1)
			h = max(0, yy2 - yy1 + 1)

			# compute the ratio of overlap between the computed
			# bounding box and the bounding box in the area list
			overlap = float(w * h) / area[j]

			# if there is sufficient overlap, suppress the
			# current bounding box
			if overlap > thres:
				suppress.append(pos)

		# delete all indexes from the index list that are in the
		# suppression list
		idxs = np.delete(idxs, suppress)

	# return only the bounding boxes that were picked
	return boxes[pick]

def drawRects(orgImg, rects, numShowRects=100):
	'''
	Helper function to draw rects on image
	:param orgImg: (np.array)
	:param rects: (np.array) shape: [num_rects, x1, y1, x2, y2]
	:param numShowRects:
	:return:
	'''
	imOut = np.array(orgImg).copy()
	ind = np.argsort(rects[:, -1])[::-1]

	# Itereate over all the region proposals
	for i in range(min(numShowRects, len(rects))):
		# draw rectangle for region proposal till numShowRects
		# draw the ones with the highest score
		rect = [int(cor) for cor in rects[ind[i]]]
		cv2.rectangle(imOut, (rect[0], rect[1]), (rect[2], rect[3]), (0, 255, 0), 1, cv2.LINE_AA)

	# show output
	plt.figure()
	plt.imshow(imOut)
	plt.show()

''' ###################################### PARAMETERS ###################################### '''

FC_STATE_DICT_PATH = '/Users/royhirsch/Documents/GitHub/DeepLearningCourse/EX2/Q1/model_params_test_loss_0.0813.pt'
FDDB_IMAGE_ORDER = '/Users/royhirsch/Documents/Study/Current/DeepLearning/Ex2/EX2_data/fddb/FDDB-folds/FDDB-fold-01.txt'
FDDB_IMAGES_ROOT = '/Users/royhirsch/Documents/Study/Current/DeepLearning/Ex2/EX2_data/fddb/images'

''' ###################################### MAIN ###################################### '''

# Load the parameters of the FC network
fc_state_dict = torch.load(FC_STATE_DICT_PATH)

# Convert state_dict of the original FC NN into FCN
'''
	How the FC-CNN was calculated ?
	after pool layer -x.shape: (128, 16, 4, 4)
	therefore if we chanel size to be 16 we need conv kernel of [output_size ,input_size=4,input_size=4]
	see: http://cs231n.github.io/convolutional-networks/#convert
'''
fcn_state_dict = fc_state_dict.copy()
fcn_state_dict['conv2.weight'] = fcn_state_dict.pop('fc1.weight').view(16, 16, 4, 4)
fcn_state_dict['conv2.bias'] = fcn_state_dict.pop('fc1.bias')

fcn_state_dict['conv3.weight'] = fcn_state_dict.pop('fc2.weight').view(2, 16, 1, 1)
fcn_state_dict['conv3.bias'] = fcn_state_dict.pop('fc2.bias')

fcn_net = Net()
fcn_net.load_state_dict(fcn_state_dict)

# Get the images file name in the right order
with open(FDDB_IMAGE_ORDER) as images_file:
	images_list = images_file.read()
	images_list = images_list.split('\n')[:-1]

# Read the images by their order
for im_name in images_list:
	full_im_path = os.path.join(FDDB_IMAGES_ROOT, im_name + '.jpg')
	im = Image.open(full_im_path)
	h_org, w_org = im.size

	# Pre-process the image
	# TODO: not sure about the resize (Roy)
	scale = 16
	im = transforms.Resize(im.size[1]/scale)(im)
	h_input, w_input = im.size
	im_tensor = transforms.ToTensor()(im).view([1, 3, w_input, h_input])

	# Evaluate the FCN
	sigmoid = nn.Sigmoid()
	output = sigmoid(fcn_net(im_tensor))

	# The score in each patch represent the score to find a face in this patch
	# class 0 stands for TRUE
	# Each neuron in score_per_patch has receptive field of 12x12 of original image
	scores = np.squeeze(output.detach().numpy())[1, :, :]
	h, w = scores.shape

	rects = scores_to_boxes(scores, 80, h_input, w_input)
	filtered_rects = non_maxima_supration(rects, thres=0.5)

	# Helper function
	drawRects(im, filtered_rects)

	# TODO missing: (Roy)
	#  - multiple scales
	#  - how to choose final rects
	#  - how to define number of faces
	#  - from rect to ellipse cordinates

	# report results to text file
	# with open('testDiscROC.txt.txt', 'w') as text_file:
	# 	text_file.write(im_name)

