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
from torch.autograd import Variable

''' ###################################### Classes ###################################### '''
class Net24(nn.Module):
    def __init__(self):
        super(Net24, self).__init__()
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

class NetFCN12(nn.Module):
    def __init__(self):
        super(NetFCN12, self).__init__()
        # [3,3] kernel ,output chanel: 16
        self.conv = nn.Conv2d(3, 16, 3)
        self.pool = nn.MaxPool2d((3, 3), stride=2)
        self.conv2 = nn.Conv2d(16, 16, 4)
        self.conv3 = nn.Conv2d(16, 2, 1)

    def forward(self, x):
        x = self.conv(x)
        x = self.pool(F.relu(x))  # x.shape = (128, 16, 4, 4)

        x = F.relu(self.conv2(x))
        x = self.conv3(x)
        return x


''' ###################################### Functions ###################################### '''

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

def scores_to_boxes(score_per_patch, thres_percentile, h_input, w_input):
    '''
    Converts output scors from the NN to rects.
    Convert each cordinate from: [x, y, h, w] to [x1, y1, x2, y2]
    :param score_per_patch: (np.array)
    :param thres_percentile: min value for score to be observed as positive, based on a percentile
    :return:
    '''
    h, w = np.shape(score_per_patch)

    # filer by score bigger then percentile
    m = np.percentile(score_per_patch.flatten(), thres_percentile)

    # Create array that holds the two points represent rectangle - top left and bottom right
    # Assume the location (i, j) in the output map is the top left corner at the input image at: (2i, 2j)
    # np.where returns a tupal of indexes 0 dim ix the x (row) ind and dim 1 is y ind (col)
    # top_left_ind.shape = [2, num_of_points]

    # Assume (i, j) is top left corner
    top_left_ind = np.array(np.where(score_per_patch >= m))
    top_left_ind = top_left_ind * 2
    bottom_right_ind = top_left_ind + np.full(top_left_ind.shape, 12)
    filter_scores = score_per_patch[np.where(score_per_patch >= m)]
    boxes = np.column_stack((top_left_ind.T, bottom_right_ind.T, filter_scores))

    return boxes


def non_maxima_supration(boxes, thres=0.5):
    '''
    same as in Q2 - maybe change according to changes there

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
        for pos in range(0, last):
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

            #  compute the IOU between rect I and rect J
            overlap = float(w * h) / (area[j] + area[i] - w * h)

            # if there is sufficient overlap, suppress the
            # current bounding box
            if overlap > thres and scores[j] <= scores[i]:
                suppress.append(pos)
            elif overlap > thres and scores[j] > scores[i]:
                del pick[-1]
                break  # found a rect with a better fit then the one being tested

        idxs = np.delete(idxs, suppress)

    # return only the bounding boxes that were picked
    return boxes[pick]


def drawRects(orgImg, rects, numShowRects=100):
    '''
    probably un-used here

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

def get_image_patches(image_tensor, boxes):
    """""
    gets a full image tensor and candidats for rects in a 2D structures:
    each row is a candidate and first 4 columns are coordinates and 5th column is the score by the net
    returns list of size number of rows and in each idx is the patch from the image after scaling it to 24*24
    """""
    if len(boxes) == 0:
        return []
    patches = []
    for box in boxes.round().long():
        # index image tensor by y, then x (row, col) - guess its true
        patch = image_tensor[:, box[1]:box[3], box[0]:box[2]]
        patch = transforms.ToPILImage()(patch)
        patch = transforms.Resize((24, 24))(patch)
        patch = transforms.ToTensor()(patch)
        patches.append(patch)
    return patches

''' ###################################### PARAMETERS ###################################### '''

FC_STATE_DICT_PATH = 'C:/Users/nimro/Desktop/ex2_local/model_params_test_loss_0.0427.pt'
net24_STATE_DICT_PATH = 'C:/Users/nimro/Desktop/ex2_local/model_q_3_params_test_loss_0.3459.pt'

FDDB_IMAGE_ORDER = 'C:/Users/nimro/Desktop/ex2_local/EX2_data/EX2_data/fddb/FDDB-folds/FDDB-fold-01.txt'
FDDB_IMAGES_ROOT = 'C:/Users/nimro/Desktop/ex2_local/EX2_data/EX2_data/fddb/images'

FDDB_IMAGE_folder = 'C:/Users/nimro/Desktop/ex2_local/EX2_data/EX2_data/fddb'
SCALES_LIST = [6, 8, 10, 12, 14, 16, 18]

net24_state_dict = torch.load(net24_STATE_DICT_PATH)

net24 = Net24()
net24.load_state_dict(net24_state_dict)

if torch.cuda.is_available():
    net24 = net24.cuda()

fc_state_dict = torch.load(FC_STATE_DICT_PATH)
fcn_state_dict = fc_state_dict.copy()
fcn_state_dict['conv2.weight'] = fcn_state_dict.pop('fc1.weight').view(16, 16, 4, 4)
fcn_state_dict['conv2.bias'] = fcn_state_dict.pop('fc1.bias')

fcn_state_dict['conv3.weight'] = fcn_state_dict.pop('fc2.weight').view(2, 16, 1, 1)
fcn_state_dict['conv3.bias'] = fcn_state_dict.pop('fc2.bias')

net12 = NetFCN12()
net12.load_state_dict(fcn_state_dict)
sigmoid = nn.Sigmoid()

threshold = 0.7 # consider change later

''' ###################################### MAIN ###################################### '''


# Get the images file name in the right order
with open(FDDB_IMAGE_ORDER) as images_file:
    images_list = images_file.read()
    images_list = images_list.split('\n')[:-1]

sigmoid = nn.Sigmoid()

# Read the images by their order
print_res_list = []
for im_name in images_list:
    print('Process image {}'.format(im_name))
    full_im_path = os.path.join(FDDB_IMAGES_ROOT, im_name + '.jpg')
    im = Image.open(full_im_path)
    image_tensor = transforms.ToTensor()(im)

    # Convert gray scale input into 3 channel input (shape is [1, h, w])
    if im.layers == 1:
        im_3_layers = torch.stack((image_tensor, image_tensor, image_tensor), dim=1) # now it is [1, 3, h, w]
        size_3_layers = im_3_layers.size()
        im_3_layers = im_3_layers.view(size_3_layers[1], size_3_layers[2], size_3_layers[3])
        im = transforms.ToPILImage()(im_3_layers)
        image_tensor = transforms.ToTensor()(im)

    h_org, w_org = im.size
    image_size_t = torch.Tensor([im.size[0], im.size[1]])

    # detect candidates for faces in different scales
    scaled_orig_rects = []
    for scale in SCALES_LIST:
        dest_size = (image_size_t / scale).round()
        real_scale = image_size_t / dest_size
        dest_size_lst = list(dest_size.numpy())
        scaled_im = transforms.Resize(dest_size_lst)(im)
        h_input, w_input = scaled_im.size

        if any(dest_size < 12):
            print("image too small for given scale: {}, {}, {}".format(scale, image_size_t, dest_size))
            continue

        im_tensor = transforms.ToTensor()(scaled_im).view([1, 3, w_input, h_input])


        # Evaluate the FCN

        output = sigmoid(net12(im_tensor))

        # The score in each patch represent the score to find a face in this patch
        # class 0 stands for TRUE
        # Each neuron in score_per_patch has receptive field of 12x12 of original image
        
        scores = np.squeeze(output.detach().numpy())[1, :, :]
        h, w = scores.shape

        rects = scores_to_boxes(scores, 60, h_input, w_input) #output [N, 5] each row is (x, y, h ,w, score)
        # N = number of detections

        # this is in comment because we want only global NMS
        #filtered_rects = non_maxima_supration(rects, thres=0.5)
        filtered_rects = rects

        # scaled_orig_rects.append(np.column_stack((filtered_rects[:, 0] * real_scale[1],
        #                                           filtered_rects[:, 2] * real_scale[1],
        #                                           filtered_rects[:, 1] * real_scale[0],
        #                                           filtered_rects[:, 3] * real_scale[0],
        #                                           filtered_rects[:, -1]))) #make sure that x is first

        scaled_orig_rects.append(np.column_stack((filtered_rects[:, 0:2] * real_scale[0],
                                                  filtered_rects[:, 2:4] * real_scale[1],
                                                  filtered_rects[:, -1])))  # make sure that x is first

    all_rects = np.concatenate(scaled_orig_rects) # there is probably a bug in the rescale

    all_rects_tensor = torch.tensor(all_rects) # N rects of different sizes
    patches = get_image_patches(image_tensor, all_rects_tensor)  # want patches of the image to enter the 24net
    # it should be a list of len=N and in each place there is an image patch as tensor
    # of shape (3, 24, 24)

    patches_batch = torch.stack(patches)
    scores = net24(Variable(patches_batch)) # scores shape = (N, 2)

    rows = []
    for i in range(scores.size()[0]):
        if scores[i][0] >  scores[i][1]: # assuming 1 - face and 0 non-face
            rows.append(i)
    all_rects_filtered = np.delete(all_rects, rows, axis=0) #ndarray of (N, 5)
    # patches__filtered = get_image_patches(image_tensor, torch.tensor(all_rects_filtered))
    all_rects_filtered = non_maxima_supration(all_rects_filtered, thres=0.5) # global nms over all candidates for rects
    patches__filtered = get_image_patches(image_tensor, torch.tensor(all_rects_filtered))
    # we should consider a lower threshold for higher recall

    # all_rects = non_maxima_supration(all_rects, thres=0.5)

    elipses = rects_to_elipses(all_rects_filtered)
    print('Num of rects: {}'.format(len(all_rects_filtered)))

    # Prepare data to print
    elipse_str_list = []

    # Convert the elipse coordinated of all the examples to str
    for num in range(len(elipses)):
        sample = elipses[num]
        elipse_str_list.append('{} {} {} {} {}  {}'.format(sample[0], sample[1], sample[2], sample[3], sample[4], sample[5]))

    # Create a list of all the values to print out
    print_res_list.append([str(im_name), str(len(elipses)), elipse_str_list])

# Report results to text file
with open('fold-01-out-after24net_check.txt', 'w') as text_file:
    for sample in print_res_list:
        text_file.write(sample[0])
        text_file.write('\n')
        text_file.write(sample[1])
        text_file.write('\n')
        for num in range(len(sample[2])):
            text_file.write(sample[2][num])
            text_file.write('\n')