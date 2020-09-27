import os
from tqdm import tqdm
import pandas as pd
from multiprocessing import Pool
import multiprocessing
from torchvision.io import read_video,write_video
import numpy as np
import torch

def read_and_reshape(inpt):
	video_dir,new_name= inpt

	width = height = 320
	sample = 10

	vframes, aframes, info = read_video(video_dir)
	t,w,h,c = vframes.shape
	w_start = int(w/2-width)
	w_end = int(w/2+width)
	h_start = int(h/2)-height
	h_end = int(h/2)+height

	vframes = vframes[:,w_start:w_end,h_start:h_end,:]
	sub_sampled = []
	for i, elem in enumerate(vframes):
		if i % sample == 0:
			sub_sampled.append(elem)
	sub_sampled = torch.stack(sub_sampled)

	write_video(new_name,sub_sampled,fps=sub_sampled.shape[0])

def load_data_from_dir(dirs,targets,target_names):
	for target in targets:
		if not os.path.exists(target):
			os.makedirs(target)

	print(f"Running on {multiprocessing.cpu_count()} cpu's.")


	p=Pool(processes=multiprocessing.cpu_count())
	max_ = len(targets)
	with tqdm(total=max_) as pbar:
		for i, _ in enumerate(p.imap_unordered(read_and_reshape, zip(dirs,target_names))):
			pbar.update()






root_dir = "../mnt/NAS/Dataset from 0915"
class_names = os.listdir(root_dir)
target_dir = "Processed_data"
directories_to_load = []
save_targets = []
save_names = []
save_labels = []

df = pd.DataFrame()
for i, class_name in enumerate(class_names):
	class_path = os.path.join(root_dir,class_name)
	sub_classes=os.listdir(class_path)
	for a_class in sub_classes:
		sub_class_true_path = os.path.join(class_path, a_class)
		for file in os.listdir(sub_class_true_path):
			directories_to_load.append(os.path.join(sub_class_true_path,file))
			save_targets.append(os.path.join(target_dir,class_name))
			save_labels.append(i)
			save_names.append(os.path.join(target_dir,class_name)+"/"+a_class+"_"+file[:-3]+"mp4")

df["data"]=save_names
df["label"]=save_labels

load_data_from_dir(directories_to_load, save_targets,save_names)

df.to_csv(target_dir+"/label_data_mapping.csv")