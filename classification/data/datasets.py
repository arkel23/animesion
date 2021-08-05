import os
import pandas as pd
from PIL import Image

import torch
import torch.utils.data as data
from torchvision import transforms

def data_loading(args, split):
    if args.model_name == 'resnet18' or args.model_name == 'resnet152':
        if split=='train':
            transform = transforms.Compose([
                transforms.Resize((args.image_size+32, args.image_size+32)),
                transforms.RandomCrop((args.image_size, args.image_size)),
                transforms.RandomHorizontalFlip(),
                transforms.ColorJitter(brightness=0.1,
                                       contrast=0.1, saturation=0.1, hue=0.1),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
            ])
        else:
            transform = transforms.Compose([
                transforms.Resize((args.image_size, args.image_size)),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
                ])
    else:
        transform = None

    if args.dataset_name == 'moeImouto':    
        dataset = moeImouto(root=args.dataset_path,
        input_size=args.image_size, split=split, transform=transform)
    elif args.dataset_name == 'danbooruFaces':
        dataset = danbooruFaces(root=args.dataset_path,
        input_size=args.image_size, split=split, transform=transform)
    elif args.dataset_name == 'cartoonFace':
        dataset = cartoonFace(root=args.dataset_path,
        input_size=args.image_size, split=split, transform=transform)
    elif args.dataset_name == 'danbooruFull':
        dataset = danbooruFull(root=args.dataset_path,
        input_size=args.image_size, split=split, transform=transform)   
    
    dataset_loader = data.DataLoader(dataset, batch_size=args.batch_size, 
        shuffle=True, num_workers=args.no_cpu_workers, drop_last=True)

    return dataset, dataset_loader


def get_transform(split, input_size):
	if split == 'train':
		transform = transforms.Compose([
				transforms.Resize((input_size+32, input_size+32)),
				transforms.RandomCrop((input_size, input_size)),
				transforms.RandomHorizontalFlip(),
				transforms.ColorJitter(brightness=0.1, 
				contrast=0.1, saturation=0.1, hue=0.1),
				transforms.ToTensor(),
				transforms.Normalize(mean=[0.5, 0.5, 0.5],
									std=[0.5, 0.5, 0.5])
				])
	else:
		transform = transforms.Compose([
				transforms.Resize((input_size, input_size)), 
				transforms.ToTensor(),
				transforms.Normalize(mean=[0.5, 0.5, 0.5],
								std=[0.5, 0.5, 0.5])
				])
	return transform


class moeImouto(data.Dataset):
	'''
	https://www.kaggle.com/mylesoneill/tagged-anime-illustrations/home
	http://www.nurs.or.jp/~nagadomi/animeface-character-dataset/
	https://github.com/nagadomi/lbpcascade_animeface
	'''
	def __init__(self, root, input_size=224, 
	split='train', transform=None):
		super().__init__()
		self.root = os.path.abspath(root)
		self.input_size = input_size
		self.split = split
		self.transform = transform

		if self.split=='train':
			print('Train set')
			self.set_dir = os.path.join(self.root, 'train.csv')
			if self.transform is None:
				self.transform = get_transform(split='train', input_size=self.input_size)			
		else:
			print('Test set')
			self.set_dir = os.path.join(self.root, 'test.csv')
			if self.transform is None:
				self.transform = get_transform(split='test', input_size=self.input_size)

		self.df = pd.read_csv(self.set_dir, sep=',', header=None, names=['class_id', 'dir'], 
			dtype={'class_id': 'UInt16', 'dir': 'object'})	

		self.targets = self.df['class_id'].to_numpy()
		self.data = self.df['dir'].to_numpy()
		
		self.classes = pd.read_csv(os.path.join(self.root, 'classid_classname.csv'), 
		sep=',', header=None, names=['class_id', 'class_name'], 
		dtype={'class_id': 'UInt16', 'class_name': 'object'})
		self.num_classes = len(self.classes)
		

	def __getitem__(self, idx):
		
		if torch.is_tensor(idx):
			idx = idx.tolist()

		img_dir, target = self.data[idx], self.targets[idx]
		img_dir = os.path.join(self.root, 'data', img_dir)
		img = Image.open(img_dir)

		if self.transform:
			img = self.transform(img)

		return img, target

	def __len__(self):
		return len(self.targets)


class danbooruFaces(data.Dataset):
	'''
	https://github.com/arkel23/Danbooru2018AnimeCharacterRecognitionDataset_Revamped
	'''
	def __init__(self, root, input_size=224, 
	split='train', transform=None):
		super().__init__()
		self.root = os.path.abspath(root)
		self.input_size = input_size
		self.split = split
		self.transform = transform

		if self.split=='train':
			print('Train set')
			self.set_dir = os.path.join(self.root, 'train.csv')
			if self.transform is None:
				self.transform = get_transform(split='train', input_size=self.input_size)

		elif self.split=='val':
			print('Validation set')
			self.set_dir = os.path.join(self.root, 'val.csv')
			if self.transform is None:
				self.transform = get_transform(split='test', input_size=self.input_size)

		else:
			print('Test set')
			self.set_dir = os.path.join(self.root, 'test.csv')
			if self.transform is None:
				self.transform = get_transform(split='test', input_size=self.input_size)

		self.df = pd.read_csv(self.set_dir, sep=',', header=None, names=['class_id', 'dir'], 
			dtype={'class_id': 'UInt16', 'dir': 'object'})

		self.targets = self.df['class_id'].to_numpy()
		self.data = self.df['dir'].to_numpy()
		
		self.classes = pd.read_csv(os.path.join(self.root, 'classid_classname.csv'), 
		sep=',', header=None, names=['class_id', 'class_name'], 
		dtype={'class_id': 'UInt16', 'class_name': 'object'})
		self.num_classes = len(self.classes)
		
	def __getitem__(self, idx):
		
		if torch.is_tensor(idx):
			idx = idx.tolist()

		img_dir, target = self.data[idx], self.targets[idx]
		img_dir = os.path.join(self.root, 'data', img_dir)
		img = Image.open(img_dir)

		if self.transform:
			img = self.transform(img)

		return img, target

	def __len__(self):
		return len(self.targets)


class danbooruFull(data.Dataset):
	'''
	https://github.com/arkel23/Danbooru2018AnimeCharacterRecognitionDataset_Revamped
	'''
	def __init__(self, root, input_size=224, 
	split='train', transform=None):
		super().__init__()
		self.root = os.path.abspath(root)
		self.input_size = input_size
		self.split = split
		self.transform = transform

		if self.split=='train':
			print('Train set')
			self.set_dir = os.path.join(self.root, 'dafre', 'train.csv')
			if self.transform is None:
				self.transform = get_transform(split='train', input_size=self.input_size)
				
		elif self.split=='val':
			print('Validation set')
			self.set_dir = os.path.join(self.root, 'dafre', 'val.csv')
			if self.transform is None:
				self.transform = get_transform(split='test', input_size=self.input_size)	
		else:
			print('Test set')
			self.set_dir = os.path.join(self.root, 'dafre', 'test.csv')
			if self.transform is None:
				self.transform = get_transform(split='test', input_size=self.input_size)

		self.df = pd.read_csv(self.set_dir, sep=',', header=None, names=['class_id', 'dir'], 
			dtype={'class_id': 'UInt16', 'dir': 'object'})

		self.targets = self.df['class_id'].to_numpy()
		self.data = self.df['dir'].to_numpy()
		
		self.classes = pd.read_csv(os.path.join(self.root, 'dafre', 'classid_classname.csv'), 
		sep=',', header=None, names=['class_id', 'class_name'], 
		dtype={'class_id': 'UInt16', 'class_name': 'object'})
		self.num_classes = len(self.classes)
		
	def __getitem__(self, idx):
		
		if torch.is_tensor(idx):
			idx = idx.tolist()

		img_dir, target = self.data[idx], self.targets[idx]
		img_dir = os.path.join(self.root, 'fullMin256', img_dir)
		img = Image.open(img_dir)
		if img.mode != 'RGB':
			img = img.convert('RGB')

		if self.transform:
			img = self.transform(img)

		return img, target

	def __len__(self):
		return len(self.targets)


class cartoonFace(data.Dataset):
	'''
	http://challenge.ai.iqiyi.com/detail?raceId=5def69ace9fcf68aef76a75d
	https://github.com/luxiangju-PersonAI/iCartoonFace
	'''
	def __init__(self, root, input_size=128, 
	split='train', transform=None):
		super().__init__()
		self.root = os.path.abspath(root)
		self.input_size = input_size
		self.split = split
		self.transform = transform

		if self.split=='train':
			print('Train set')
			self.set_dir = os.path.join(self.root, 'train.csv')
			if self.transform is None:
				self.transform = get_transform(split='train', input_size=self.input_size)
			
		else:
			print('Test set')
			self.set_dir = os.path.join(self.root, 'test.csv')
			if self.transform is None:
				self.transform = get_transform(split='test', input_size=self.input_size)

		self.df = pd.read_csv(self.set_dir, sep=',', header=None, names=['class_id', 'dir'], 
			dtype={'class_id': 'UInt16', 'dir': 'object'})

		self.targets = self.df['class_id'].to_numpy()
		self.data = self.df['dir'].to_numpy()
		
		self.classes = pd.read_csv(os.path.join(self.root, 'classid_classname.csv'), 
		sep=',', header=None, names=['class_id', 'class_name'], 
		dtype={'class_id': 'UInt16', 'class_name': 'object'})
		self.num_classes = len(self.classes)
		

	def __getitem__(self, idx):
		
		if torch.is_tensor(idx):
			idx = idx.tolist()

		img_dir, target = self.data[idx], self.targets[idx]
		img_dir = os.path.join(self.root, 'data', img_dir)
		img = Image.open(img_dir)

		if self.transform:
			img = self.transform(img)

		return img, target

	def __len__(self):
		return len(self.targets)
