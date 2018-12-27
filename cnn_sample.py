#作者：LZD
#项目：Pytorch_First
#时间：2018/12/26 18:28
# -*- coding:utf-8 -*-


import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
from collections import OrderedDict
from torch.autograd import Variable


# 0.建立数据
x = torch.unsqueeze(torch.linspace(-1, 1, 100), dim=1) # x data (tensor), shape=(100, 1)
y = x.pow(2)+0.2*torch.rand(x.size())


# #用Variable来修饰这些数据tensor
# x, y = torch.autograd.Variable(x), Variable(y)
#
#
# #画图
# plt.scatter(x.data.numpy(), y.data.numpy(), c='r', marker=".")
# plt.show()



# 1.建立网络
# class Net(torch.nn.Module):  #继承torch的Moudle
# 	def __int__(self, n_feature, n_hidden, n_output):
# 	#def __int__(self):
# 		super(Net, self).__int__()   #继承__int__功能
# 		#定义每层用什么样的形式
# 		self.hidden = torch.nn.Linear(1, 10) #隐藏层线性输出
# 		self.predict = torch.nn.Linear(10, 1) #输出层线性输出
#
# 	def forward(self, x): #这同时也是Module中的forward功能
# 		# 正向传播输入值，神经网络分析输出值
# 		x = F.relu(self.hidden(x))   #激励函数（隐藏层的线性值）
# 		x = self.predict(x)          #输出值
# 		return x
# class Net(torch.nn.Module):
# 	def __init__(self, fea, hid, out):
# 		super(Net, self).__init__()
# 		self.hidden = torch.nn.Linear(fea, hid)  # hidden layer
# 		self.predict = torch.nn.Linear(hid, out)  # output layer
#
#
# 	def forward(self, x):
# 		x = F.relu(self.hidden(x))  # activation function for hidden layer
# 		x = self.predict(x)  # linear output
# 		return x

class Net(torch.nn.Module):
	def __init__(self, fea, hid, out):
		super(Net, self).__init__()
		self.conv=torch.nn.Sequential()
		self.conv.add_module("one", torch.nn.Linear(fea, hid))
		self.conv.add_module("two", torch.nn.Linear(hid, out))



	def forward(self, x):
		x = F.relu(self.conv.one(x))  # activation function for hidden layer
		print("one=", x)
		x = self.conv.two(x)  # linear output
		print("two", x)
		return x


net = Net(fea=1,hid=10,out=1)
#net = Net()
print(net)  #net的结构

"""
Net(
	(hidden):Liner(1->10)
	(predict):Liner(10->1)
	)
"""


# 2.训练网络
#optimizer是训练的工具
optimizer = torch.optim.SGD(net.parameters(), lr=0.5)  #传入net的所有参数， 学习率为0.5
loss_func = torch.nn.MSELoss()  #预测值和真实值的误差计算公式（均方差）


for t in range(500):
	prediction = net(x)  #传给net训练数据x，输出预测值
	loss = loss_func(prediction, y)  #计算两者的误差
	optimizer.zero_grad()  #清除上一步的残差更新参数值
	loss.backward()        #误差反向传播，计算参数更新值
	optimizer.step()       #将参数更新值施加到net的parameter上
	if t%5 == 0:
		plt.cla()
		plt.scatter(x.data.numpy(), y.data.numpy())
		plt.plot(x.data.numpy(), prediction.data.numpy(), 'r-', lw=5)
		plt.text(0.5, 0, 'Loss=%.4f' % loss.data.numpy(), fontdict={'size':20, 'color':'red'})
		plt.pause(0.1)

# 3.可视化训练过程
plt.ioff()  #画图
plt.show()

