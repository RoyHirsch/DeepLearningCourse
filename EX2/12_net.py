import torch
import torchfile
from torch.utils.data import Dataset, DataLoader
import os
import numpy as np
from PIL import Image
from torchvision import transforms

aflw_path = 'C:/Users/dorim/Desktop/DOR/TAU uni/Msc/DL/EX2/EX2_data/aflw'
filename = 'aflw_12.t7'
pascal_path = 'C:/Users/dorim/Desktop/DOR/TAU uni/Msc/DL/EX2/EX2_data/VOCdevkit/VOC2007'

class aflw_loader(Dataset):
    'loads the 12*12 images as a num_sampels * dimension numpy and prepars'
    'a labels vector of size num_sampels  and samples a tensor of each'
    def __init__(self, path,filename):
        self.rawdata = torchfile.load(os.path.join(path, filename),force_8bytes_long=True)
        self.rawdata = np.array(list(self.rawdata.values()))
        self.rawdata  = self.rawdata.reshape((self.rawdata.shape[0],self.rawdata.shape[1]*
                               self.rawdata.shape[2]*self.rawdata.shape[3]))
        self.labels = np.ones((self.rawdata.shape[0]))

    def __len__(self):
        return len(self.rawdata)

    def __getitem__(self, idx):
        sample = self.rawdata[idx]
        sample_labels = self.labels[idx]
        norm_sample = sample_labels.reshape([1, 12, 12])
        return torch.tensor(norm_sample).float(), torch.tensor(sample_labels).float()



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
    labels = np.zeros((num_images))
    return pascal_as_list,labels

class Pascal_loader(Dataset):
    'Input: pascal images not containing a person as a list of numpy arrays, negative labels as a numpy array'
    'Output: A normalized random sized croppings of images as tensors and their labels as tensors'
    def __init__(self,full_images,labels):
        self.data = full_images
        self.labels = labels
        self.transform = transforms.Compose([transforms.RandomSizedCrop(12),transforms.ToTensor()])
    def __len__(self):
        return len(self.data)
    def __getitem__(self, idx):
        sample = self.data[idx]
        norm_sample = np.fromstring(sample, sep=' ') / 255.
        sample_labels = self.labels[idx]
        if self.transform:
            sample_tensors = self.transform(norm_sample)
        return sample_tensors.float(), torch.tensor(sample_labels).float()

positive_aflw_12_net = aflw_loader(path = aflw_path, filename = filename)
pascal_images,pascal_labels = load_pascal_to_numpy(pascal_path)
negative_pascal_12_net = Pascal_loader(full_images = pascal_images, labels = pascal_labels)

batch_size_pos = 4
batch_size_neg = 12

train_loader = DataLoader(aflw_loader, batch_size=batch_size_pos)
test_loader = DataLoader(Pascal_loader, batch_size=batch_size_neg)
