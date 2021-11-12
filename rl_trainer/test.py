import torch
import torch.nn as nn
import os
class Mycnn(nn.Module):
	def __init__(self):
		super(Mycnn, self).__init__()   # 继承__init__功能
		# 第一层卷积
		self.conv1 = torch.nn.Sequential(  # input_size = 25*25*1
			torch.nn.Conv2d(in_channels=1, out_channels=12, kernel_size=9, stride=1, padding=4),
			torch.nn.ReLU(),
			torch.nn.MaxPool2d(kernel_size=2, stride=2)  # output_size = 12*12*12
		)
		self.conv2 = torch.nn.Sequential(  # input_size = 12*12*12
			torch.nn.Conv2d(12, 24, 5, 1, 2),
			torch.nn.ReLU(),
			torch.nn.MaxPool2d(2, 2)  # output_size = 6*6*24
		)
		self.conv3 = torch.nn.Sequential(  # input_size = 6*6*24
			torch.nn.Conv2d(24, 48, 3, 1, 1),
			torch.nn.ReLU(),
			torch.nn.MaxPool2d(2, 2)  # output_size = 3*3*48
		)
		self.dense = torch.nn.Sequential(
			torch.nn.Linear(432, 216),
			torch.nn.ReLU(),
			torch.nn.Dropout(0.5),
			torch.nn.Linear(216, 108),
			torch.nn.ReLU(),
			torch.nn.Dropout(0.5),
			torch.nn.Linear(108, 59)
		)
	def forward(self, x):   #正向传播过程
		conv1_out = self.conv1(x.to(torch.float32))
		conv2_out = self.conv2(conv1_out)
		conv3_out = self.conv3(conv2_out)
		'''
		x.view(x.size(0), -1)的用法：
		在CNN中，因为卷积或者池化之后需要连接全连接层，所以需要把多维度的tensor展平成一维，因此用它来实现(其实就是将多维数据展平为一维数据方便后面的全连接层处理)
		'''
		res = conv3_out.view(conv3_out.size(0), -1)
		out = self.dense(res)
		return out
model=Mycnn()
modelpath=os.path.dirname(os.path.abspath(__file__)) +"/models/cnnmodels2.pth"
model=torch.load(modelpath)
torch.save(model.state_dict(),modelpath)