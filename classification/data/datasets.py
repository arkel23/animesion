import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import torch
import torch.utils.data as data
from torchvision import transforms, datasets

class moeImoutoDataset():
	# most images are between 100 to 200 pixels square
	# resample to 128 for convenience to make same as danbooruFacesCrops
	def __init__(self, input_size=128,
	data_path='/home2/edwin_ed520/personal/moeimouto_animefacecharacterdataset/'):
		self.data_path = os.path.abspath(data_path)
		self.input_size = input_size

		self.data_transform = transforms.Compose([
        transforms.Resize((self.input_size, self.input_size)),
        transforms.RandomHorizontalFlip(),
		transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1, hue=0.1),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5],
                             std=[0.5, 0.5, 0.5])
    	])

	def getImageFolder(self):
		self.dataset = datasets.ImageFolder(root=self.data_path, 
		transform=self.data_transform)
		return self.dataset

class danbooruFacesCrops(data.Dataset):
	# this dataset is just the crops of the faces to 128x128
	def __init__(self, input_size=128,
	base_folder="/home2/edwin_ed520/personal/Danbooru2018AnimeCharacterRecognitionDataset/"):
		super().__init__()
		self.input_size = input_size
		self.img_folder = os.path.join(base_folder, "danbooru2018_animefacecropdataset")
		
		# contains names of class (character names) assigned to certain ID. ex: 
		# miyu_(vampire_princess_miyu)	0
		self.class_names_dir = os.path.join(base_folder, "tagIds.tsv")
		self.class_names_pd = pd.read_csv(self.class_names_dir, sep='\t',
		header=None, names=['character', 'id'])
		self.class_names = self.class_names_pd.to_numpy()
		
		# first col is the img path, second is class id, third col is detection results
		# Each detection result has five fields separated by commas, 
		# i.e. left, top, right, bottom, confidence in order. ex:
		#  0446\332446.jpg	1	0.448534, 0.441995, 0.584106, 0.654420 0.337403
		self.data_dic_dir = os.path.join(base_folder, "faces.tsv")
		self.data_dic_pd = pd.read_csv(self.data_dic_dir, sep='\t',
		header=None, names=['dir', 'id', 'coords'])
		self.data_dic = self.data_dic_pd.to_numpy()
	
	def __len__(self):
		return len(self.data_dic)
	
	'''
	def __getitem__(self, index):
		# Select sample
		ID = self.list_IDs[index]

    	# Load data and get label
		X = torch.load('data/' + ID + '.pt')
		y = self.labels[ID]
		
		return X, y
	'''

	def no_classes(self):
		return len(self.class_names)

	def stats(self):
		calc_stats(self.data_dic_pd.copy(), self.no_classes(), self.class_names)


def calc_stats(samples_per_class, no_classes, class_names):
	samples_per_class = samples_per_class.groupby('id', as_index=True).count()['dir'].squeeze()
		
	set_mean = samples_per_class.mean()
	set_median = samples_per_class.median()
	set_std = samples_per_class.std()
	print('Dataset no of classes: {}\n'
	'Mean number of samples per class: {}\n'
	'Median number of samples per class: {}\n'
	'Standard deviation of samples per class: {}'.format(
	no_classes, set_mean, set_median, set_std))

	samples_per_class_ordered = samples_per_class.sort_values(0, ascending=False)
	print(samples_per_class_ordered.head())

	print('Characters with most number of samples: ')
	print('\t'.join('No. {}: {} (Class ID: {}) with {} samples'.format(
    j+1, class_names[samples_per_class_ordered.index[j], 0], 
	class_names[samples_per_class_ordered.index[j], 1], 
	samples_per_class_ordered.iloc[j]) for j in range(10)))

	print('\nCharacters with least number of samples: ')
	print('\t'.join('No. {}: {} (Class ID: {}) with {} samples'.format(
    j+1, class_names[samples_per_class_ordered.index[-1-j], 0], 
	class_names[samples_per_class_ordered.index[-1-j], 1], 
	samples_per_class_ordered.iloc[-1-j]) for j in range(10)))

	fig, axs = plt.subplots(1)
	fig.suptitle('Histogram of Classes for DanbooruFace Dataset')

	# only plot first 100
	bins = 100
	axs.bar(np.arange(bins), samples_per_class_ordered.iloc[0:bins].to_numpy())
		
	axs.set_ylabel('No. of samples per class')
	axs.set_title('Ordered based on no. of samples')
		
	results_dir = 'results'
	if not os.path.exists(results_dir):
		os.makedirs(results_dir)
	fig.savefig(os.path.join(results_dir, 
	'histogram_danbooruFacesCrops.png'), dpi=300)
		