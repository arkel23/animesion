import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from PIL import Image

import torch
import torch.utils.data as data
from torchvision import transforms, datasets

from utils.dataset_stats import calc_stats

class moeImoutoDataset():
	# most images are between 100 to 200 pixels square
	# resample to 128 for convenience to make same as danbooruFacesCrops
	def __init__(self, input_size=128,
	data_path=r'C:\Users\ED520\edwin\data\moeimouto_animefacecharacterdataset',
	transform=transforms.Compose([
        transforms.Resize((128, 128)),
        transforms.RandomHorizontalFlip(),
		transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1, hue=0.1),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5],
                             std=[0.5, 0.5, 0.5])
    	])):
		self.data_path = os.path.abspath(data_path)
		self.input_size = input_size
		self.transform = transform

	def getImageFolder(self):
		self.dataset = datasets.ImageFolder(root=self.data_path, 
		transform=self.transform)
		return self.dataset

class danbooruFacesCrops(data.Dataset):
	# this dataset is just the crops of the faces to 128x128
	def __init__(self, split, input_size=128,
	base_folder="/home2/edwin_ed520/personal/Danbooru2018AnimeCharacterRecognitionDataset/", 
	transform=None):
		super().__init__()
		self.input_size = input_size
		self.split = split
		self.img_folder = os.path.join(base_folder, "danbooru2018_animefacecropdataset")
		self.transform = transform
		if self.transform is not None:
			self.transform = transforms.Compose([
			transforms.Resize((self.input_size, self.input_size)),
			transforms.RandomHorizontalFlip(),
			transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1, hue=0.1),
			transforms.ToTensor(),
			transforms.Normalize(mean=[0.5, 0.5, 0.5],
								std=[0.5, 0.5, 0.5])
			])

		# contains names of class (character names) assigned to certain ID. ex: 
		# miyu_(vampire_princess_miyu)	0
		self.class_names_dir = os.path.join(base_folder, "tagIds.tsv")
		self.class_names_pd = pd.read_csv(self.class_names_dir, sep='\t',
		header=None, names=['character', 'class_id'])
		self.class_names = self.class_names_pd.to_numpy()
		
		# first col is the img path, second is class id, third col is detection results
		# added a fourth col for img id (img path without jpg and the subfolder)
		# Each detection result has five fields separated by commas, 
		# i.e. left, top, right, bottom, confidence in order. ex:
		#  0446\332446.jpg	1	0.448534, 0.441995, 0.584106, 0.654420 0.337403
		self.data_dic_dir = os.path.join(base_folder, "faces_mod.tsv")
		self.data_dic_pd = pd.read_csv(self.data_dic_dir, sep='\t',
		header=None, names=['dir', 'class_id', 'coords', 'file_id'],
		dtype={'dir': 'object', 'class_id': 'int64', 'coords': 'object', 'file_id': 'object'})
		self.data_dic = self.data_dic_pd.to_numpy()

		if self.split=='train':
			# train split data
			self.set_list_dir = os.path.join(base_folder, "trainSplit.tsv")
			self.set_list = pd.read_csv(self.set_list_dir, sep='\t', 
			header=None, names=['file_id'], dtype={'file_id': 'object'})
			print('Initialized {} split'.format(split))
		elif self.split=='val':
			# validation split data
			self.set_list_dir = os.path.join(base_folder, "valSplit.tsv")
			self.set_list = pd.read_csv(self.set_list_dir, sep='\t', 
			header=None, names=['file_id'], dtype={'file_id': 'object'})
			print('Initialized {} split'.format(split))
		elif self.split=='test':
			# test split data
			self.set_list_dir = os.path.join(base_folder, "testSplit.tsv")
			self.set_list = pd.read_csv(self.set_list_dir, sep='\t', 
			header=None, names=['file_id'], dtype={'file_id': 'object'})
			print('Initialized {} split'.format(split))

		# for compatibility with standard pytorch sets
		# classes and class_to_idx atribute
		self.classes = self.class_names[:, 0]
		self.class_to_idx = dict(zip(self.class_names[:, 0], self.class_names[:, 1]))
		# samples and targets attributes
		# samples is a list of tuple of two values [(img_dir, label)]
		# targets is a list of [label]
		self.subset_df = self.data_dic_pd.loc[self.data_dic_pd['file_id'].isin(self.set_list['file_id'].to_numpy())]
		self.targets = self.subset_df['class_id'].values.tolist()
		self.subset_image_dirs = self.subset_df['dir'].values.tolist()
		self.samples = list(zip(self.subset_image_dirs, self.targets))

	def __len__(self):
		return len(self.set_list)
	
	def __getitem__(self, idx):
		
		if torch.is_tensor(idx):
			idx = idx.tolist()

		# get one file_id from the train/val/test split
		key_id = self.set_list['file_id'].iloc[idx]
		# get the row of df based on the given file_id
		row = self.data_dic_pd.loc[self.data_dic_pd['file_id'] == key_id]
		# get the dir and class_id columns
		img_dir, label = row[['dir', 'class_id']].iloc[0]
		# split the dir into the folder and the file name
		folder, file_name = img_dir[:4], img_dir[5:]
		# concatenate full path and read img
		img_dir = os.path.join(self.img_folder, folder, file_name)
		image = Image.open(img_dir)
		
		if self.transform:
			image = self.transform(image)

		return image, label

	def no_classes(self):
		return len(self.class_names)

	def stats(self):
		# for original dataset with 900 k images
		calc_stats(self.data_dic_pd.copy(), self.no_classes(), self.class_names)
		# for subset with 85% confidence (500k images)
		print(self.class_names_pd['class_id'].nunique())
		print(self.data_dic_pd['class_id'].nunique())
		classes_reduced = self.subset_df['class_id'].nunique()
		print(self.subset_df.head())
		print(classes_reduced)
		#calc_stats(self.subset_df.copy(), classes_reduced, self.class_names)
		# total no of classes is 70k
		# main label file only includes 49k
		# in training 34k
		# in val 3.6k
		# assume training, val and test dont add up to 49k
		# one alternative is to make a new subset of ids and
		# traininng, val and test lists into only one
		# then when needed to split just split like [0.8], [0.1]
		# alternative is to use the 49 k even if no images in certain classes
		# good is that when move on to the danbooru512x512 not many
		# changes are needed, only change the main root directory
		# and split baased on the heuristics suggested 0.8, 0.1, 0.1
		# another alternative is to only include ids with more 
		# than mean no of samples 
