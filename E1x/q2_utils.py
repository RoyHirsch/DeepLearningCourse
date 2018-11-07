import numpy as np
import torch

def oneHotEncoder(data, nb_classes):
	targets = np.array(data).reshape(-1)
	return np.eye(nb_classes)[targets]

def randData(data, label):
	comb = np.hstack([data, label.reshape(-1,1)])
	comb = np.random.permutation(comb)
	return torch.tensor(comb[:,:-1]).float(), torch.tensor(comb[:,-1]).long()

def getBatch(data, label, batch_size, i):
	return data[i:i+batch_size], label[i:i+batch_size]
