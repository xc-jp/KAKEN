import torch
from src.Network import custom_conv_layer
from src.Network import custom_deconv_layer

class Encoder(torch.nn.Module):
	def __init__(self):
		super(Encoder, self).__init__()
		self.conv1 = custom_conv_layer(padding=[0,0,0], in_channels=3, out_channels=16, kernel_size=[3,3,3], stride = [2,2,2]) #161
		self.conv2 = custom_conv_layer(padding=[0,0,0], in_channels=16, out_channels=32, kernel_size=[3,3,3], stride = [2,2,2]) #80
		self.conv3 = custom_conv_layer(padding=[0,0,0], in_channels=32, out_channels=8, kernel_size=[3,3,3], stride = [2,2,2]) #80
		self.linear = torch.nn.Linear(800,500)


		self.relu = torch.nn.ReLU()
		self.sigmoid = torch.nn.Sigmoid()

		self.out_shape = []



	def forward(self, x, print_shapes = False):
		shapes = ""

		try: 
			shapes+=str(x.shape)+"\n"
			x=self.conv1(x)

			shapes+=str(x.shape)+"\n"
			x=self.conv2(x)

			shapes+=str(x.shape)+"\n"
			x=self.conv3(x)

			shapes+=str(x.shape)+"\n"
			self.out_shape = x.shape
			x=self.linear(x.view(x.shape[0],-1))
			x=self.sigmoid(x)
			shapes+=str(x.shape)+"\n"

			return x

		finally:
			if print_shapes:
				print(shapes)

class Decoder(torch.nn.Module):
	def __init__(self):
		super(Decoder, self).__init__()
		self.linear = torch.nn.Linear(500,800)

		self.conv3 = custom_deconv_layer(padding=[0,0,0], in_channels=16, out_channels=3, kernel_size=[4,4,4], stride = [4,4,4],use_activation=False) #161
		self.conv2 = custom_deconv_layer(padding=[2,2,0], in_channels=32, out_channels=16, kernel_size=[2,2,2], stride = [4,4,2]) #80
		self.conv1 = custom_deconv_layer(padding=[2,2,0], in_channels=8, out_channels=32, kernel_size=[2,2,2], stride = [4,4,2]) #80


		self.relu = torch.nn.ReLU()
		self.sigmoid = torch.nn.Sigmoid()

		self.out_shape = []



	def forward(self, x, out_shape, print_shapes = False):
		shapes = ""

		try: 
			shapes+=str(x.shape)+"\n"

			x=self.linear(x).view(out_shape)
			x=self.relu(x)

			shapes+=str(x.shape)+"\n"
			x=self.conv1(x)

			shapes+=str(x.shape)+"\n"
			x=self.conv2(x)

			shapes+=str(x.shape)+"\n"
			x=self.conv3(x)
			x=self.relu(x)
			shapes+=str(x.shape)+"\n"

			return x

		finally:
			if print_shapes:
				print(shapes)


class EncoderDecoder(torch.nn.Module):
	def __init__(self):
		super(EncoderDecoder,self).__init__()
		self.encoder = Encoder()
		self.decoder = Decoder()

	def forward(self, x):
		en_out = self.encoder.forward(x)
		de_out = self.decoder.forward(en_out, out_shape=self.encoder.out_shape)
		return de_out

def test():
	encoderdecoder = EncoderDecoder()
	loss_fun = torch.nn.MSELoss()
	test_tensor=torch.rand([1, 3, 640, 640, 16])
	optimizer = torch.optim.Adam(encoderdecoder.parameters())
	for i in range(100):
		de_out=encoderdecoder(test_tensor)
		optimizer.zero_grad()
		loss=loss_fun(de_out,test_tensor).sum()
		loss.backward()
		optimizer.step()
		print(loss.item())

if __name__ == "__main__":
	test()