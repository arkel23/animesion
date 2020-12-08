import os

from torchvision import transforms, datasets

class moeImoutoDataset():
	def __init__(self, input_size=112,
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

