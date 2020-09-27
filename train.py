from src.Network import dummy_convnet
from src.DataLoader import HairMicroscopeDataset
from torch.utils.data import DataLoader
import torch
import numpy as np
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter
import argparse
from torch.utils.data import random_split
from torch.nn.functional import normalize
def __main__():
	parser = argparse.ArgumentParser()
	parser.add_argument('--data-dir', default="./Processed_data/")
	parser.add_argument('--log-dir', default="./")
	args = parser.parse_args()
	dataset_root = args.data_dir
	net = dummy_convnet().to("cuda")
	loss_fun=torch.nn.NLLLoss()
	optimizer = torch.optim.Adam(net.parameters())
	ds = HairMicroscopeDataset(dataset_root)
	split_ratio = int(len(ds)*70/100)
	train,test=random_split(ds,[split_ratio,len(ds)-split_ratio])
	dataloader_train = DataLoader(train, batch_size=10,shuffle=True, num_workers=8)
	dataloader_test = DataLoader(test, batch_size=8,shuffle=True, num_workers=8)

	summarywriter=SummaryWriter("./logs/"+args.log_dir)


	for epoc in tqdm(range(100)):
		loss_arr=[]
		accuracy = []

		for i_batch, sample_batched in tqdm(enumerate(dataloader_train)):

			data, label = sample_batched["data"].to("cuda").float(), sample_batched["label"].unsqueeze(-1).to("cuda").float()
			data=normalize(data, p=2, dim=1, eps=1e-12, out=None)
			preds = net(data)
			loss=loss_fun(preds.float(),label.squeeze().long())

			optimizer.zero_grad()
			loss.backward()
			optimizer.step()
			loss_arr.append(loss.cpu().item())

			max_indices=torch.argmax(preds.detach(),dim=1)
			correct_guesses = np.sum([ 1 for pred, l in zip(max_indices.long(), label.long()) if pred==l])
			accuracy.append((correct_guesses/float(len(label))))

		summarywriter.add_scalar("epoc loss",np.mean(loss_arr),epoc)
		summarywriter.add_scalar("accuracy",np.mean(accuracy),epoc)

		if epoc %1==0:
			loss_arr=[]
			accuracy = []
			with torch.no_grad():
				for i_batch, sample_batched in tqdm(enumerate(dataloader_test)):
					data, label = sample_batched["data"].to("cuda").float(), sample_batched["label"].unsqueeze(-1).to("cuda").float()
					data=normalize(data, p=2, dim=1, eps=1e-12, out=None)

					preds = net(data)
					loss=loss_fun(preds.float(),label.squeeze().long())

					loss_arr.append(loss.cpu().item())

					max_indices=torch.argmax(preds.detach(),dim=1)
					correct_guesses = np.sum([ 1 for pred, l in zip(max_indices.long(), label.long()) if pred==l])
					accuracy.append((correct_guesses/float(len(label))))

			summarywriter.add_scalar("epoc loss test",np.mean(loss_arr),epoc)
			summarywriter.add_scalar("accuracy test",np.mean(accuracy),epoc)

	summarywriter.close()
if __name__=="__main__":

	__main__()
