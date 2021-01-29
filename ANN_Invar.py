# -*- coding: utf-8 -*-
"""
Created on Fri Jul 17 13:34:00 2020

@author: z.rao
"""


# import csv
# import numpy as np
# with open('D:/experiments/ML project of Invar alloys/Code for ANN/Data base.xlsx','r') as csvfile:
#     reader = csv.reader(csvfile)
#     rows= [row for row in reader]
# print (rows)#输出所有数据
# data=np.array(rows)#rows是数据类型是‘list',转化为数组类型好处理
# print("out0=",type(data),data.shape)
# print("out1=",data)
import sys
import os
import datetime
import torch.utils.data as Data
import pandas as pd
import torch
import torch.nn.functional as F     # 激励函数都在这
import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import train_test_split
import torch.nn as nn
import time
# import shap
import seaborn as sns
from My_Modules import MAPELoss
from My_Modules import atomic_properties
# import xgboost
# from torch.utils.tensorboard import SummaryWriter   

plt.close('all')
starttime = datetime.datetime.now()

data = pd.read_excel('Data_base.xlsx')
df_test = data.drop(columns=['alloy'])

# writer = SummaryWriter('log')


# x1 = df_test [['Fe','Ni','Co','Cr','V','Cu']]
# x1 = df_test [['Fe','Ni','Co','Cr','V','Cu','VEC']]
# x1 = df_test [['Fe','Ni','Co','Cr','V','Cu','NM']]
# x1 = df_test [['Fe','Ni','Co','Cr','V','Cu']]
x1 = df_test [['Fe','Ni','Co','Cr','V','Cu']]


x2 = df_test [['VEC','AR1','AR2','PE','Density','TC','MP','FI','SI','TI','M']]


min = [8,135,124,1.8235,7874,75.905,1181,737.14,1562.98,2957.4,0.6]
max = [10,140,125.679,1.91,8908,116.559,1850.4,762.47,1753.03,3395,2.22]
min = np.array(min)
max = np.array(max)

# x2 = df_test [['VEC','M']]


x2_normalization = (x2 - min)/(max-min)

x_combined = pd.concat ((x1, x2_normalization), axis=1)

y1 = df_test [['TEC']]


x = torch.FloatTensor(x_combined.values).cuda()
y = torch.FloatTensor(y1.values).cuda()
# x = torch.FloatTensor(x1.values).cuda() 
# y = torch.FloatTensor(y1.values).cuda()
  # noisy y data (tensor), shape=(100, 1)

# print (x.shape)

# named_x = x.refine_names('Fe', 'Ni')

# print(x,named_x)

train_features, test_features, train_labels, test_labels = train_test_split(x, 
                                                                            y, 
                                                                            test_size=0.15,
                                                                            random_state=36)

train_features = train_features
train_labels = train_labels
test_features = test_features
test_labels = test_labels

print(train_features.device)
print(train_labels.device)
print(test_features.device)
print(test_labels.device)


batch_size = 16

print(train_features)
# print(x)
# print(y)
# plt.plot(x , y, "ro")
# plt.show()

class Net(nn.Module):  # 继承 torch 的 Module
    def __init__(self, n_feature=17, n_hidden=218, n_output=1, w = 6,):
        super(Net, self).__init__()     # 继承 __init__ 功能
        # 定义每层用什么样的形式
        # self.BN=torch.nn.BatchNorm1d(n_hidden)
        self.hidden1 = torch.nn.Linear(n_feature, n_hidden) 
        nn.init.kaiming_normal_(self.hidden1.weight)
        
        self.hiddens = nn.ModuleList ([nn.Linear(n_hidden, n_hidden) for i in range(w)])                            
        for m in self.hiddens:
            nn.init.kaiming_normal_(m.weight)   
        
        self.predict = torch.nn.Linear(n_hidden, n_output)   # 输出层线性输出
        nn.init.kaiming_normal_(self.predict.weight)

    def forward(self, x):   # 这同时也是 Module 中的 forward 功能
        # 正向传播输入值, 神经网络分析出输出值
        x = self.hidden1(x)
        # x = self.BN(x)
        # x = self.Dropout (x)
        x = F.relu(x)   
        
        for m in self.hiddens:
            x = m(x)
            # x = self.BN(x)
            x = F.relu(x) 
                      
        x = self.predict(x)
        # x = self.BN_3(x)
        # x = self.Dropout (x)
          # 输出值
        return x


net = Net()

net.cuda()

print(net)  # net 的结构】

print(next(net.parameters()).device)



optimizer = torch.optim.Adam(net.parameters(), lr=0.0005,weight_decay=0.0001)  # 传入 net 的所有参数, 学习率
loss = MAPELoss() 
# loss = loss.cuda()     # 预测值和真实值的误差计算公式 (均方差)
# loss = MAPELoss() 
num_epochs=5000

model_name = 'ANN_Invar_36'
t = time.localtime()
folder_dir = 'Results/{:02}-{:02}-{:02}-{:02}-{}'.format(t[1], t[2], t[3], t[4], model_name)
if not os.path.isdir(folder_dir):
    os.mkdir(folder_dir)

def plotCurve(x_vals, y_vals, 
                x_label, y_label, 
                x2_vals=None, y2_vals=None, 
                legend=None,
                figsize=(3.5, 2.5)):
    # set figsize
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.plot(x_vals, y_vals)
    if x2_vals and y2_vals:
        plt.plot(x2_vals, y2_vals, linestyle=':')
    
    if legend:
        plt.legend(legend)
        
def train(net, num_epochs, batch_size,
          train_features, train_labels, 
          test_features, test_labels,
          loss, optimizer):
    print ("=== train begin ===")
    # data process
    train_dataset = Data.TensorDataset(train_features, train_labels)
    test_dataset = Data.TensorDataset(test_features, test_labels)
    train_iter = Data.DataLoader(train_dataset, batch_size, shuffle=True)
    test_iter = Data.DataLoader(test_dataset, batch_size, shuffle=True)
    # test_iter = Data.DataLoader(test_dataset, shuffle=True)
    
    # train by step
    train_ls, test_ls = [], []
    for epoch in range(num_epochs):
        for x, y in train_iter:
            ls = loss(net(x).view(-1, 1), y.view(-1, 1))
            # writer.add_scalar('train loss', ls, global_step=None, walltime=None)
            optimizer.zero_grad()
            ls.backward()
            optimizer.step()
        
        # save loss for each step
        train_ls.append(loss(net(train_features).view(-1, 1), train_labels.view(-1, 1)).item())
        test_ls.append(loss(net(test_features).view(-1, 1), test_labels.view(-1, 1)).item())
    
        if epoch % 10 == 0:
            print ("epoch %d: train loss %f, test loss %f" % (epoch, train_ls[-1], test_ls[-1]))
            save_model_dir = str(model_name + "_trainloss_" + "{}".format(train_ls[-1]) 
                                 + "_testloss_" + "{}".format(test_ls[-1]) + "_{}.pth".format(epoch))
            torch.save(net.state_dict(), os.path.join(folder_dir, save_model_dir))
    
    print ("final epoch: train loss %f, test loss %f" % (train_ls[-1], test_ls[-1]))
    
    print ("plot curves")
    plotCurve(range(1, num_epochs + 1), train_ls,
              "epoch", "loss",
              range(1, num_epochs + 1), test_ls,
              ["train", "test"]
              )
    plt.text(1500, 0.8, 'Loss=%.4f' % test_ls[-1], fontdict={'size': 20, 'color':  'red'})
    plt.savefig('Figures/{:02}-{:02}-{:02}-{:02}-{}_loss.png'.format(t[1], t[2], t[3], t[4], 
                                                                       model_name),index=False, format='png', dpi=300)
    print ("=== train end ===")

    
train(net, num_epochs, batch_size, train_features, train_labels, test_features, test_labels, loss, optimizer)

net.load_state_dict(torch.load('Results/01-27-17-11-ANN_Invar_36/ANN_Invar_36_trainloss_0.040329642593860626_testloss_0.08329400420188904_4960.pth'))
net.eval()

# Q17_19 = pd.read_excel('Q17-19.xlsx')
# Q17_19 = Q17_19.drop(columns=['alloy'])
# x_17_1 = Q17_19 [['Fe','Ni','Co','Cr','V','Cu']]
# x_17_2 = Q17_19 [['VEC','AR1','AR2','PE','Density','TC','MP','FI','SI','TI','M']]
# x_17_2_normalization = (x_17_2 - min)/(max - min)
# x_17_combined = pd.concat ((x_17_1, x_17_2_normalization), axis=1)
# x_17 = torch.FloatTensor(x_17_combined.values).cuda()

# predict_17 = net(x_17).cpu()
# predict_17 = predict_17.data.numpy()

predict=net(test_features).cpu()
predict=predict.data.numpy()

train_predict=net(train_features).cpu()
train_predict=train_predict.data.numpy()


# # Fe_train=x_train [:,0]
# # Fe_test=x_test [:,0]
# # print(loss)

plt.figure()
# # # plt.scatter(Fe_train.numpy().reshape(-1,1), y_train.numpy(),color='blue')
# # # plt.scatter(Fe_test.numpy(), predict, color='red')
# # # plt.scatter(Fe_test.numpy().reshape(-1,1), y_test.numpy(), color='blue')
# plt.scatter(test_labels.numpy().reshape(-1,1), predict, color='red')
# plt.axis([0, 20, 0, 20])
sns.set()
test_labels = test_labels
test_labels = test_labels.cpu()
sns.regplot(x=test_labels.numpy().reshape(-1,1), y=predict, robust=True, color='g')  
# plt.savefig('D:\\experiments\\ML project of Invar alloys\\Code for ANN\\results\\14_32BS_6elements_4_16hidden_MAPEloss_Adam_Ir0.0005_epoch_5000\\17_2（Data_base）.png', format='png', dpi=300)
ax = plt.subplot()

train_labels = train_labels.cpu()

test_features_cpu = test_features.cpu()
x_test_features = test_features_cpu.numpy()
train_features_cpu = train_features.cpu()
x_train_features = train_features_cpu.numpy()

x_test_labels=test_labels.numpy().reshape(-1,1)
x_train_labels=train_labels.numpy().reshape(-1,1)
ax.scatter(x=train_labels.numpy().reshape(-1,1), y=train_predict, c='black', alpha=0.6)  # 改变颜色
ax.scatter(x=test_labels.numpy().reshape(-1,1), y=predict, c='red',alpha=0.6)
# ax.scatter(x_train_labels[204,0],train_predict[204,0], c='red', alpha=0.5)
# plt.savefig('Figures/{:02}-{:02}-{:02}-{:02}-{}_predict.png'.format(t[1], t[2], t[3], t[4], 
                                                                       # model_name),index=False, format='png', dpi=300)

# explainer = shap.DeepExplainer(net,train_features)
# shap_values = explainer.shap_values(train_features)
train_bias = np.abs(x_train_labels - train_predict)
test_bias = np.abs(x_test_labels - predict)

train_combined = np.concatenate((x_train_features,x_train_labels,train_predict,train_bias), axis=1)
train_combined = pd.DataFrame(train_combined)
train_combined.columns = ['Fe','Ni','Co','Cr','V','Cu','VEC','AR1','AR2','PE',
                          'Density','TC','MP','FI','SI','TI','M','train_labels','train_predict','bias']
# train_combined.to_excel('Results/{:02}-{:02}-{:02}-{:02}-{}_train.xlsx'.format(t[1], t[2], t[3], t[4], 
#                                                                        model_name),index=False)
test_combined = np.concatenate ((x_test_features,x_test_labels,predict,test_bias),axis=1)
test_combined = pd.DataFrame(test_combined)
test_combined.columns = ['Fe','Ni','Co','Cr','V','Cu','VEC','AR1','AR2','PE',
                          'Density','TC','MP','FI','SI','TI','M','test_labels','test_predict','bias']
# test_combined.to_excel('Results/{:02}-{:02}-{:02}-{:02}-{}_test.xlsx'.format(t[1], t[2], t[3], t[4], 
#                                                                        model_name),index=False)







matplotlib=True 
# # visualize the first prediction's explanation (use matplotlib=True to avoid Javascript)




plt.show()

# torch.save(net, 'D:\\experiments\\ML project of Invar alloys\\Code for ANN\\results\\14_32BS_6elements_4_16hidden_MAPEloss_Adam_Ir0.0005_epoch_5000\\17_（Data_base）_net.pkl')  # 保存整个网络
# torch.save(net.state_dict(), 'D:\\experiments\\ML project of Invar alloys\\Code for ANN\\results\\14_32BS_6elements_4_16hidden_MAPEloss_Adam_Ir0.0005_epoch_5000\\17_（Data_base）_net_params.pkl')   # 只保存网络中的参数 (速度快, 占内存少)
# torch.save(net.state_dict(), 'D:\\experiments\\ML project of Invar alloys\\Code for ANN\\results\\4_16BS_6elements_4_16_hidden_MAPEloss_Adam_lr0.0005_epoch_10000\\7_net_params.pt')
# net2 = torch.load('D:\\experiments\\ML project of Invar alloys\\Code for ANN\\results\\3_16BS_17elements_4_16_hidden_MAPEloss_Adam_lr0.0005_epoch_10000\\1_net.pkl')
# predict1 = net2(test_features)
# print(predict, predict1.data.numpy())

# 将保存的参数复制到 net
# net.load_state_dict(torch.load('D:\\experiments\\ML project of Invar alloys\\Code for ANN\\results\\3_16BS_17elements_4_16_hidden_MAPEloss_Adam_lr0.0005_epoch_10000\\1_net_params.pkl'))
# predict2 = net(x)
# print(predict2.data.numpy())

endtime = datetime.datetime.now()
print (endtime - starttime)



