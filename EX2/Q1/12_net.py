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


''' ###################################### PARAMETERS ###################################### '''
# aflw_path = 'C:/Users/nimro/Desktop/ex2_local/EX2_data/EX2_data/aflw'
aflw_path = '/Users/royhirsch/Documents/Study/Current/DeepLearning/Ex2/EX2_data/aflw'
filename = 'aflw_12.t7'
# pascal_path = 'C:/Users/dorim/Desktop/DOR/TAU uni/Msc/DL/EX2/EX2_data/VOCdevkit/VOC2007'
pascal_path = '/Users/royhirsch/Documents/Study/Current/DeepLearning/Ex2/VOCdevkit/VOC2007'
test_par = 0.1
dropout_par = 0.25
batch_size_pos = 16
batch_size_neg = 64
n_epoches = 200

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

class Pascal_loader(Dataset):
    'Input: pascal images not containing a person as a list of numpy arrays, negative labels as a numpy array'
    'Output: A normalized random sized croppings of images as tensors and their labels as tensors'
    def __init__(self, full_images, labels):
        self.data = full_images
        self.labels = labels
        self.transform = transforms.Compose([transforms.ToPILImage(), transforms.RandomResizedCrop(12), transforms.ToTensor()])

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        sample = self.data[idx]
        sample_labels = self.labels[idx]
        sample = self.transform(sample)
        # sample = np.swapaxes(sample, 0, 2)
        # shape: [12,12,3]
        return sample.float(), torch.tensor(sample_labels).float()

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        # [3,3] kernel ,output chanel: 16
        self.conv = nn.Conv2d(3, 16, 3)
        self.drop2D = nn.Dropout2d(p=dropout_par, inplace=False)
        self.pool = nn.MaxPool2d((3, 3), stride=2)
        self.fc1 = nn.Linear(256, 16)
        self.fc2 = nn.Linear(16, 2)
        self.relu = nn.ReLU(inplace=True)


    def forward(self, x):
        x = self.conv(x)
        x = self.drop2D(x)
        x = self.pool(x)
        x = self.relu(x)

        x = x.view(-1, 256)

        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        return x

''' ###################################### FUNCTIONS ###################################### '''
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

def permutate_input_n_labels(pos_inputs, pos_labels, neg_inputs, neg_labels):
    '''
    Gets positive and negative data and label and shuffle them,
    :param pos_inputs: (torch.tensor)
    :param pos_labels: (torch.tensor)
    :param neg_inputs: (torch.tensor)
    :param neg_labels: (torch.tensor)
    :return: shuffled data and labels
    '''
    merged_inputs = np.vstack((pos_inputs, neg_inputs))
    merged_labels = np.vstack((pos_labels, neg_labels))
    inds = list(range(len(merged_labels)))
    np.random.shuffle(inds)

    return torch.tensor(merged_inputs[inds, :, :, :]).float(), torch.tensor(merged_labels[inds, :]).flatten().long()

''' ###################################### MAIN ###################################### '''

# Create the date loaders
positive_aflw_12_net = Aflw_loader(path=aflw_path, filename=filename)
pascal_images, pascal_labels = load_pascal_to_numpy(pascal_path)
negative_pascal_12_net = Pascal_loader(full_images=pascal_images, labels = pascal_labels)

pos_train_sampler, pos_test_sampler = create_sampler_for_train_n_test(len(positive_aflw_12_net.rawdata), test_par)
neg_train_sampler, neg_test_sampler = create_sampler_for_train_n_test(len(negative_pascal_12_net.data), test_par)

pos_train_loader = DataLoader(positive_aflw_12_net, batch_size=batch_size_pos, sampler=pos_train_sampler)
neg_train_loader = DataLoader(negative_pascal_12_net, batch_size=batch_size_neg, sampler=neg_train_sampler)

pos_test_loader = DataLoader(positive_aflw_12_net, batch_size=batch_size_pos, sampler=pos_test_sampler)
neg_test_loader = DataLoader(negative_pascal_12_net, batch_size=batch_size_neg, sampler=neg_test_sampler)

# Create net
net = Net()
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(net.parameters(), lr=0.001)

train_loss_total = []
test_loss_total = []

# Main optimization loop:
for epoch in range(n_epoches):

    train_loss_arr = []
    # Because neg_train_loader iterator is smaller - use cycle iterator (Roy)
    for i, data in enumerate(zip(pos_train_loader, cycle(neg_train_loader)), 0):
        (pos_inputs, pos_labels), (neg_inputs, neg_labels) = data
        inputs, labels = permutate_input_n_labels(pos_inputs, pos_labels, neg_inputs, neg_labels)
        optimizer.zero_grad()

        net.train()
        outputs = net(inputs)
        train_loss_tmp = criterion(outputs, labels)
        train_loss_tmp.backward()
        optimizer.step()
        train_loss_arr.append(train_loss_tmp.item())

    train_loss_epoch = np.mean(train_loss_arr)
    train_loss_total.append(train_loss_epoch)

    # Test the model
    test_loss_arr = []
    for data in zip(pos_test_loader, cycle(neg_test_loader)):
        (pos_inputs, pos_labels), (neg_inputs, neg_labels) = data
        inputs, labels = permutate_input_n_labels(pos_inputs, pos_labels, neg_inputs, neg_labels)

        net.eval()
        outputs = net(inputs)
        test_loss_t = criterion(outputs, labels)
        test_loss_arr.append(test_loss_t.item())

    test_loss_epoch = np.mean(test_loss_arr)
    test_loss_total.append(test_loss_epoch)

    print('\n==> Epoch num {} :'.format(epoch))
    print('train loss : {0:3.4f}'.format(train_loss_epoch))
    print('test loss : {0:3.4f}'.format(test_loss_epoch))

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

