import torch
import pandas as pd
import torchvision
from torch.utils.data import Dataset
import os
import numpy as np
from torchvision.transforms import RandomHorizontalFlip
class HairMicroscopeDataset(Dataset):
	"""Face Landmarks dataset."""

	def __init__(self, root_dir, transform=None):
		"""
		Args:
			csv_file (string): Path to the csv file with annotations.
			root_dir (string): Directory with all the images.
			transform (callable, optional): Optional transform to be applied
				on a sample.
		"""
		
		self.conf_file = pd.read_csv(root_dir+"label_data_mapping.csv")
		self.root_dir = root_dir
		self.transform = transform

	def __len__(self):
		return len(self.conf_file)

	def __getitem__(self, idx):
		
		if torch.is_tensor(idx):
			idx = idx.tolist()

		img_name = os.path.join(self.root_dir,
								self.conf_file["data"].iloc[idx])
		vframes,aframes,info=torchvision.io.read_video(img_name)
		vframes = vframes.float()
		# noise=torch.randn(vframes.shape)
		# vframes+=noise

		# if np.random.random() > 0.5:
		# 	vframes=torch.flip(vframes, [1]) 

		# if np.random.random() > 0.5:
		# 	vframes=torch.flip(vframes, [2]) 

		# if self.transform:
		#     arr = []

		#     for frame in vframes:
		#         print(frame.shape)
		#         arr.append(self.transform(frame))
		#     vframes = torch.vstack(arr)
		vframes=vframes.permute([3,1,2,0]).float()
		labels = self.conf_file["label"].iloc[idx]
		sample = {'data': vframes, 'label': labels}
		

		return sample

