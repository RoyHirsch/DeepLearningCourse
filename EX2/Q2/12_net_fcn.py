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

''' ###################################### CLASSES ###################################### '''

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        # [3,3] kernel ,output chanel: 16
        self.conv = nn.Conv2d(3, 16, 3)
        self.pool = nn.MaxPool2d((3, 3), stride=2)
        self.conv2 = nn.Conv2d(16, 16, 4)
        self.conv3 = nn.Conv2d(16, 2, 1)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.conv(x)
        x = self.pool(x)        # x.shape = (128, 16, 4, 4)
        x = self.relu(x)

        x = self.conv2(x)
        x = self.relu(x)
        x = self.conv3(x)
        return x

''' ###################################### FUNCTIONS ###################################### '''

def rects_to_elipses(rects):
	'''

	Transforms the rect to elipse coordinates format.
	<major_axis_radius minor_axis_radius angle center_x center_y 1>.
	:param square: rects
	:return: elipses
	'''

	# transform rect representation from [x1,y1,x2,y2] to [h, w, angle, center_x, center_y] elipse cords
	center_x = (rects[:, 2] + rects[:, 0]) / 2
	center_y = (rects[:, 3] + rects[:, 1]) / 2
	w = rects[:, 2] - rects[:, 0]
	h = (rects[:, 2] - rects[:, 0]) * 1.2
	angle = np.zeros_like(h)

	major_axis_radius = h / math.sqrt(2)
	minor_axis_radius = w / math.sqrt(2)

	elipses = np.column_stack((major_axis_radius, minor_axis_radius, angle, center_x, center_y, rects[:,-1]))

	return elipses

def scores_to_boxes(score_per_patch, threshold, h_input, w_input):
	'''
	Converts output scors from the NN to rects.
	Convert each cordinate from: [x, y, h, w] to [x1, y1, x2, y2]
	:param score_per_patch: (np.array)
	:param threshold: min score (from Sigmoid)
	:return:
	'''
	h, w = np.shape(score_per_patch)

	# filer by score bigger then percentile

	# Create array that holds the two points represent rectangle - top left and bottom right
	# Assume the location (i, j) in the output map is the top left corner at the input image at: (2i, 2j)
	# np.where returns a tupal of indexes 0 dim ix the x (row) ind and dim 1 is y ind (col)
	# top_left_ind.shape = [2, num_of_points]

	# Assume (i, j) is top left corner
	top_left_ind = np.array(np.where(score_per_patch >= threshold))
	top_left_ind = top_left_ind * 2
	bottom_right_ind = top_left_ind + np.full(top_left_ind.shape, 12)
	filter_scores = score_per_patch[np.where(score_per_patch >= threshold)]
	boxes = np.column_stack((top_left_ind.T, bottom_right_ind.T, filter_scores))

	return boxes

def non_maxima_supration(boxes, thres=0.5):
	'''
	The output scores image of the backbone NN
	:param score_per_patch: (np.array)
	:param thres_percentile: min value for score to be observed as positive, based on a percentile
	:param thres: threshold for IOU of overlaping rects
	:return:
	'''
	pick = []

	x1 = boxes[:, 0]
	y1 = boxes[:, 1]
	x2 = boxes[:, 2]
	y2 = boxes[:, 3]
	scores = boxes[:, 4]

	area = (x2 - x1 + 1) * (y2 - y1 + 1)
	idxs = np.argsort(y2)

	while len(idxs) > 0:

		last = len(idxs) - 1
		i = idxs[last]
		pick.append(i)
		suppress = [last]

		for pos in range(0, last):
			j = idxs[pos]

			xx1 = max(x1[i], x1[j])
			yy1 = max(y1[i], y1[j])
			xx2 = min(x2[i], x2[j])
			yy2 = min(y2[i], y2[j])

			w = max(0, xx2 - xx1 + 1)
			h = max(0, yy2 - yy1 + 1)

			#  compute the IOU between rect I and rect J
			overlap = float(w * h) / (area[j] + area[i] - w * h)

			# if there is sufficient overlap, suppress the
			# current bounding box
			if overlap > thres and scores[j] <= scores[i]:
				suppress.append(pos)

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

FC_STATE_DICT_PATH = '/Users/royhirsch/Documents/GitHub/DeepLearningCourse/EX2/Q1/model_params_test_loss_0.0427.pt'
FDDB_IMAGE_ORDER = '/Users/royhirsch/Documents/Study/Current/DeepLearning/Ex2/EX2_data/fddb/FDDB-folds/FDDB-fold-01.txt'
FDDB_IMAGES_ROOT = '/Users/royhirsch/Documents/Study/Current/DeepLearning/Ex2/EX2_data/fddb/images'
SCALES_LIST      = [3, 4, 5, 6, 7, 8, 10, 12, 14, 16, 18, 19, 20]

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

# Get the images file name in the right order
with open(FDDB_IMAGE_ORDER) as images_file:
	images_list = images_file.read()
	images_list = images_list.split('\n')[:-1]

# Read the images by their order
print_res_list = []
sigmoid = nn.Sigmoid()
for im_name in images_list:
	print('Process image {}'.format(im_name))
	full_im_path = os.path.join(FDDB_IMAGES_ROOT, im_name + '.jpg')
	im = Image.open(full_im_path)
	h_org, w_org = im.size

	scaled_orig_rects = []
	for scale in SCALES_LIST:
		scaled_im = transforms.Resize(int(im.size[1]/scale))(im)
		h_input, w_input = scaled_im.size

		# Convert gray scale input into 3 channel input
		if im.layers == 1:
			scaled_im = np.dstack((scaled_im, scaled_im, scaled_im))
		im_tensor = transforms.ToTensor()(scaled_im).view([1, 3, w_input, h_input])

		# Evaluate the FCN
		output = sigmoid(fcn_net(im_tensor))

		# The score in each patch represent the score to find a face in this patch
		# class 0 stands for TRUE
		# Each neuron in score_per_patch has receptive field of 12x12 of original image
		scores = np.squeeze(output.detach().numpy())[1, :, :]
		h, w = scores.shape

		rects = scores_to_boxes(scores, 0.1, h_input, w_input)
		filtered_rects = non_maxima_supration(rects, thres=0.5)
		scaled_orig_rects.append(np.column_stack((filtered_rects[:, :-1] * scale, filtered_rects[:, -1])))

	all_rects = np.concatenate(scaled_orig_rects)

	elipses = rects_to_elipses(all_rects)
	print('Num of rects: {}'.format(len(all_rects)))

	# Prepare data to print
	elipse_str_list = []

	# Convert the elipse coordinated of all the examples to str
	for num in range(len(elipses)):
		sample = elipses[num]
		elipse_str_list.append('{} {} {} {} {} {}'.format(sample[0], sample[1], sample[2], sample[3], sample[4], sample[5]))

	# Create a list of all the values to print out
	print_res_list.append([str(im_name), str(len(elipses)), elipse_str_list])

# Report results to text file
with open('fold-01-out.txt', 'w') as text_file:
	for sample in print_res_list:
		text_file.write(sample[0])
		text_file.write('\n')
		text_file.write(sample[1])
		text_file.write('\n')
		for num in range(len(sample[2])):
			text_file.write(sample[2][num])
			text_file.write('\n')
