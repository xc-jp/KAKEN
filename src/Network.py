import torch
class custom_conv_layer(torch.nn.Module):

    def __init__(self,padding,in_channels,out_channels,kernel_size,stride):
        super(custom_conv_layer,self).__init__()
        self.conv_1 = torch.nn.Conv3d(padding=padding, in_channels=in_channels,out_channels=out_channels,kernel_size=kernel_size, stride = stride)
        if out_channels>1:
            self.batchnorm = torch.nn.BatchNorm3d(out_channels)
        else:
            self.batchnorm = None
            
        self.activation = torch.nn.ReLU()
        
        
    def forward(self,x):
        x=self.conv_1(x)
        
        if not self.batchnorm == None:
            x=self.batchnorm(x)
            
        x=self.activation(x)
        return x

class custom_deconv_layer(torch.nn.Module):

    def __init__(self,padding,in_channels,out_channels,kernel_size,stride,use_activation=True):
        super(custom_deconv_layer,self).__init__()
        self.conv_1 = torch.nn.ConvTranspose3d(output_padding=padding,in_channels=in_channels,out_channels=out_channels,kernel_size=kernel_size, stride = stride)
        if out_channels>1:
            self.batchnorm = torch.nn.BatchNorm3d(out_channels)
        else:
            self.batchnorm = None
            
        self.use_activation = use_activation
        self.activation = torch.nn.ReLU()
        
        
    def forward(self,x):
        x=self.conv_1(x)
        
        if not self.batchnorm == None:
            x=self.batchnorm(x)

        if self.use_activation:
            x=self.activation(x)
        return x
        

class dummy_convnet(torch.nn.Module):

    def __init__(self):
        super(dummy_convnet, self).__init__()
        self.conv_1 = custom_conv_layer(padding=[0,0,0], in_channels=3,out_channels=16,kernel_size=[3,3,3], stride = [1,1,1])
        self.conv_2 = custom_conv_layer(padding=[0,0,0], in_channels=16,out_channels=32,kernel_size=[1,1,1], stride = [2,2,2])
        self.conv_3 = custom_conv_layer(padding=[1,1,1], in_channels=32,out_channels=64,kernel_size=[3,3,3], stride = [1,1,1])        
        self.conv_4 = custom_conv_layer(padding=[0,0,0], in_channels=64,out_channels=32,kernel_size=[2,2,2], stride = [2,2,2])
        self.conv_5 = custom_conv_layer(padding=[0,0,0], in_channels=32,out_channels=16,kernel_size=[2,2,2], stride = [2,2,2])
        self.conv_6 = custom_conv_layer(padding=[0,0,0], in_channels=16,out_channels=1,kernel_size=[1,2,1], stride = [20,20,1])
        self.dense = torch.nn.Linear(16,2)
        self.softmax=torch.nn.Softmax(dim=-1)
        
    def forward(self,x):
        x=self.conv_1(x)
        # print(x.shape)
        x=self.conv_2(x)

        # print(x.shape)

        x=self.conv_3(x)

        # print(x.shape)

        x=self.conv_4(x)

        # print(x.shape)

        x=self.conv_5(x)

        # print(x.shape)

        x=self.conv_6(x)
        # print(x.shape)

        x=self.dense(x.view(x.shape[0],-1))
        x=self.softmax(x)
        # print(x.shape)

        return x

