import os
import torch

import torch
import torch.nn as nn


import numpy as np

device = 'cpu'


actions_map = {0: [-100, -30], 1: [-100, -18], 2: [-100, -6], 3: [-100, 6], 4: [-100, 18], 5: [-100, 30], 6: [-40, -30],
               7: [-40, -18], 8: [-40, -6], 9: [-40, 6], 10: [-40, 18], 11: [-40, 30], 12: [20, -30], 13: [20, -18],
               14: [20, -6], 15: [20, 6], 16: [20, 18], 17: [20, 30], 18: [80, -30], 19: [80, -18], 20: [80, -6],
               21: [80, 6], 22: [80, 18], 23: [80, 30], 24: [140, -30], 25: [140, -18], 26: [140, -6], 27: [140, 6],
               28: [140, 18], 29: [140, 30], 30: [200, -30], 31: [200, -18], 32: [200, -6], 33: [200, 6], 34: [200, 18],
               35: [200, 30]}           #dicretise action space


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
modelpath=os.path.dirname(os.path.abspath(__file__)) +"/cnnmodels1.pth"
model.load_state_dict(torch.load(modelpath))


def my_controller(observation_list, action_space_list, is_act_continuous):
	obs = observation_list['obs']
	obs = obs[np.newaxis, np.newaxis, :, :]  # (1,1,25,25)
	obs = torch.tensor(obs)
	action_prob = model(obs)
	actions_raw = torch.max(action_prob, 1)[1].data.numpy().squeeze().item()
	actions = actions_map[actions_raw]
	wrapped_actions = [[actions[0]], [actions[1]]]
	return wrapped_actions

