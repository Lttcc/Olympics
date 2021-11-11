import os

import numpy as np
from abc import abstractmethod
import torch
import torch.nn as nn
import pickle

def load_obj(name ):
    with open('data/map4/' + name + '.pkl', 'rb') as f:
        return pickle.load(f)
torch.manual_seed(1)
import torch.utils.data as Data
torch.manual_seed(1)
# 设置超参数
epoches = 100
batch_size = 25
learning_rate = 0.001
class Mycnn(nn.Module):
	def __init__(self):
		super(Mycnn, self).__init__()   # 继承__init__功能
		## 第一层卷积
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
def main():
    # cnn 实例化
    cnn = Mycnn()
    print(cnn)

    # 定义优化器和损失函数
    optimizer = torch.optim.Adam(cnn.parameters(), lr=learning_rate)
    loss_function = nn.CrossEntropyLoss()
    #载入数据
    expert_data = load_obj("all_data")
    data=np.array(expert_data['obs'])
    #data=np.divide(data,255)#归一化
    # for i in range(130):
	#     for j in range(210):
	# 	    for k in range(160):
	# 		    for h in range(3):
	# 			    if data[i][j][k][h] != 0:
	# 				    print(data[i][j][k][h])
    #data=np.transpose(data,(0,3,1,2))
   # print(data.shape)
    data=np.expand_dims(data.astype(float),axis=-1)
    data=np.transpose(data,(0,3,1,2))
    data_x=torch.tensor(data)
    label=expert_data['action']
    label_y=torch.tensor(label)
    test_x=data_x
    test_y=label_y
    torch_experdata=Data.TensorDataset(data_x,label_y)
    #print(type(torch_experdata))
    train_loader=Data.DataLoader(
		dataset=torch_experdata,
	    batch_size=batch_size,
	    shuffle=True,
	    num_workers=3
    )
	# 开始训练
    for epoch in range(epoches):
        print("进行第{}个epoch".format(epoch))
        for step, (batch_x, batch_y) in enumerate(train_loader):
            output = cnn(batch_x)  # batch_x=[50,3,210,160]
            # output = output[0]
            loss = loss_function(output, batch_y)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            print("进行第{}个step".format(step))
            if step % 1 == 0:
                test_output = cnn(test_x)  # [10000 ,10]
                pred_y = torch.max(test_output, 1)[1].data.numpy()
                # accuracy = sum(pred_y==test_y)/test_y.size(0)
                accuracy = ((pred_y == test_y.data.numpy()).astype(int).sum()) / float(test_y.size(0))
            print('Epoch: ', epoch, '| train loss: %.4f' % loss.data.numpy(), '| test accuracy: %.2f' % accuracy)
        if accuracy>0.97:
            break

    path = os.getcwd()+"\\models\\"+"cnnmodels1.pth"
    torch.save(cnn, path)
    test_output = cnn(test_x[:10])
    pred_y = torch.max(test_output, 1)[1].data.numpy().squeeze()
    # t=torch.max(test_output,1)
    # print(t[0])
    print(pred_y)
    print(test_y[:10])


if __name__ == "__main__":
    main()

