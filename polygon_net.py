import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import os
import json
from torch.utils.data import Dataset, DataLoader, random_split
from torchvision.transforms.functional import rotate, hflip, vflip
from PIL import Image

curr_path = os.getcwd()
train_path = os.path.join(curr_path, 'dataset', 'training')
val_path = os.path.join(curr_path, 'dataset', 'validation')
meta_train_path = os.path.join(curr_path, 'dataset', 'training', 'data.json')
meta_val_path = os.path.join(curr_path, 'dataset', 'validation', 'data.json')

try:
	with open('encoded_color_dict.json', 'r') as f:
		encoded_color_dict = json.load(f)
except:
	raise ValueError("Please have a .json file mapping colors to their respective numerical representation.")
else:
	num_colors = len(encoded_color_dict)

try:
	with open(meta_train_path, 'r') as f:
		meta_train_info = json.load(f)
except:
	raise ValueError("Failed to load data.json containing meta information of input/output pairs for training.")

try:
	with open(meta_val_path, 'r') as f:
		meta_val_info = json.load(f)
except:
	raise ValueError("Failed to load data.json containing meta information of input/output pairs for validation.")

class PolygonDataset(Dataset):
	def __init__(self, mode, augmentation=False, seed=None):
		self._seed = seed
		self.random = np.random.RandomState(self._seed)

		self.augmentation = augmentation
		self.valid_modes = ['train', 'validate']

		if mode not in self.valid_modes:
			raise ValueError("Enter a Valid Mode!")

		self.encoded_color_dict = encoded_color_dict

		if mode == 'train':
			self.path = train_path
			self.meta_info = meta_train_info
		elif mode == 'validate':
			self.path = val_path
			self.meta_info = meta_val_info
			
	def __len__(self):
		return len(self.meta_info)

	def __getitem__(self, idx):
		polygon_img_name = self.meta_info[idx]['input_polygon']
		target_img_name = self.meta_info[idx]['output_image']

		polygon_img_path = os.path.join(self.path, 'inputs', polygon_img_name)
		target_img_path = os.path.join(self.path, 'outputs', target_img_name)

		input_color = self.meta_info[idx]['colour']
		coloridx = self.encoded_color_dict[input_color]

		polygon = Image.open(polygon_img_path)
		polygon = np.array(polygon).astype(np.float32) / 255.0        #Normalizing before sending it out

		target = Image.open(target_img_path)
		target = np.array(target).astype(np.float32) / 255.0		  #Normalizing before sending it out

		x1 = torch.from_numpy(polygon).float().permute(2, 0, 1)
		x2 = torch.tensor(coloridx).reshape(1)
		y = torch.from_numpy(target).float().permute(2, 0, 1)

		if self.augmentation:
			if self.random.uniform() > 0.5:
				angle = self.random.uniform(0, 90)
				x1 = rotate(x1, angle)
				y = rotate(y, angle)

			if self.random.uniform() > 0.5:
				x1 = hflip(x1)
				y = hflip(y)

			if self.random.uniform() > 0.5:
				x1 = vflip(x1)
				y = vflip(y)

		return (x1, x2), y


class UNet(nn.Module):
	def __init__(self, in_channels=6, out_channels=3, output_dims=(128, 128)):
		super(UNet, self).__init__()        
		self.enc1 = nn.Sequential(
			nn.Conv2d(in_channels, 32, kernel_size=3),
			nn.BatchNorm2d(32),
			nn.ReLU(),
			nn.Conv2d(32, 32, kernel_size=3),
			nn.ReLU()
		)
		self.down1 = nn.MaxPool2d(2)

		self.enc2 = nn.Sequential(
			nn.Conv2d(32, 64, kernel_size=3),
			nn.BatchNorm2d(64),
			nn.ReLU(),
			nn.Conv2d(64, 64, kernel_size=3),
			nn.ReLU()
		) 
		self.down2 = nn.MaxPool2d(2)

		self.enc3 = nn.Sequential(
			nn.Conv2d(64, 128, kernel_size=3),
			nn.BatchNorm2d(128),
			nn.ReLU(),
			nn.Conv2d(128, 128, kernel_size=3),
			nn.ReLU()
		)

		self.down3 = nn.MaxPool2d(2)

		self.bottleneck = nn.Sequential(
			nn.Conv2d(128, 253, kernel_size=3),
			nn.BatchNorm2d(253),
			nn.ReLU(),
			nn.Conv2d(253, 253, kernel_size=3),
			nn.ReLU()
		)
		self.up3 = nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2)

		self.dec3 = nn.Sequential(
			nn.Conv2d(256, 128, kernel_size=3),
			nn.BatchNorm2d(128),
			nn.ReLU(),
			nn.Conv2d(128, 128, kernel_size=3),
			nn.ReLU()
		)

		self.up2 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)

		self.dec2 = nn.Sequential(
			nn.Conv2d(128, 64, kernel_size=3),
			nn.BatchNorm2d(64),
			nn.ReLU(),
			nn.Conv2d(64, 64, kernel_size=3),
			nn.ReLU()
		)

		self.up1 = nn.ConvTranspose2d(64, 32, kernel_size=2, stride=2)

		self.dec1 = nn.Sequential(
			nn.Conv2d(64, 32, kernel_size=3),
			nn.BatchNorm2d(32),
			nn.ReLU(),
			nn.Conv2d(32, 32, kernel_size=3),
			nn.ReLU()
		)

		self.reshape = nn.AdaptiveAvgPool2d(output_dims)

		self.out = nn.Sequential(
			nn.Conv2d(32, 32, kernel_size=3, padding=1),
			nn.BatchNorm2d(32),
			nn.ReLU(),
			nn.Conv2d(32, 16, kernel_size=3, padding=1),
			nn.ReLU(),
			nn.Conv2d(16, 8, kernel_size=3, padding=1),
			nn.ReLU(),
			nn.Conv2d(8, out_channels, kernel_size=3, padding=1)
		)

	def forward(self, x, encoded_color_maps):
		x = torch.cat([x, encoded_color_maps], dim=1)
		x1 = self.enc1(x)
		x2 = self.down1(x1)
		x2 = self.enc2(x2)
		x3 = self.down2(x2)
		x3 = self.enc3(x3)
		x4 = self.down3(x3)
		
		x4 = self.bottleneck(x4)
		
		color_map = F.interpolate(encoded_color_maps, size=(8, 8), mode='bilinear', align_corners=False)
		x4 = torch.cat([x4, color_map], dim=1)

		x4 = self.up3(x4)
		x3 = F.interpolate(x3, size=(16, 16), mode='bilinear', align_corners=False)
		x4 = self.dec3(torch.cat([x4, x3], dim=1))
		x4 = self.up2(x4)
		x2 = F.interpolate(x2, size=(24, 24), mode='bilinear', align_corners=False)
		x4 = self.dec2(torch.cat([x4, x2], dim=1))
		x4 = self.up1(x4)
		x1 = F.interpolate(x1, size=(40, 40), mode='bilinear', align_corners=False)
		x4 = self.dec1(torch.cat([x4, x1], dim=1))
		
		x4 = self.reshape(x4)
		out = self.out(x4)

		return out

class PolyGonNet(nn.Module):
	def __init__(self, num_colors=num_colors, embedding_dim=3, output_dims=(128, 128), input_dims=(128, 128)):
		super(PolyGonNet, self).__init__()

		self.input_dims = input_dims
		self.Unet = UNet()
		self.encoder = nn.Embedding(num_embeddings=num_colors, embedding_dim=embedding_dim)

	def forward(self, image, coloridx):
		color_embed = self.encoder(coloridx).view(-1, 3, 1, 1).expand(-1, -1, *self.input_dims)
		result = self.Unet(image, color_embed)

		return result
		
		