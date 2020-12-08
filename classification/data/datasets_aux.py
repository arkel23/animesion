import os
import numpy as np

import torch
import torch.utils.data as data
from torchvision.datasets import CIFAR10
from torchvision import transforms, datasets

class Dataset(data.Dataset):
  'Characterizes a dataset for PyTorch'
  def __init__(self, list_IDs, labels):
        'Initialization'
        self.labels = labels
        self.list_IDs = list_IDs

  def __len__(self):
        'Denotes the total number of samples'
        return len(self.list_IDs)

  def __getitem__(self, index):
        'Generates one sample of data'
        # Select sample
        ID = self.list_IDs[index]

        # Load data and get label
        X = torch.load('data/' + ID + '.pt')
        y = self.labels[ID]

        return X, y

# modified according to TA
class CIFAR10_truncated(data.Dataset):

	def __init__(self, root, dataidxs=None, train=True, transform=None, target_transform=None, download=False):

		self.root = root
		self.dataidxs = dataidxs
		self.train = train
		self.transform = transform
		self.target_transform = target_transform
		self.download = download

		self.data, self.target = self.__build_truncated_dataset__()

	def __build_truncated_dataset__(self):

		cifar_dataobj = CIFAR10(self.root, self.train, self.transform, self.target_transform, self.download)
		if self.train is True:
			data = np.array(cifar_dataobj.data)
			target = np.array(cifar_dataobj.targets)
		else:
			data = np.array(cifar_dataobj.data)
			target = np.array(cifar_dataobj.targets)

		if self.dataidxs is not None:
			data = data[self.dataidxs]
			target = target[self.dataidxs]

		return data, target

	def __getitem__(self, index):
		"""
        Args:
            index (int): Index

        Returns:
            tuple: (image, target) where target is index of the target class.
        """
		img, target = self.data[index], self.target[index]

		if self.transform is not None:
			img = self.transform(img)

		if self.target_transform is not None:
			target = self.target_transform(target)

		return img, target

	def __len__(self):
		return len(self.data)
