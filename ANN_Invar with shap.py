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
import torch.utils.data as Data
import pandas as pd
import torch
import torch.nn.functional as F     # 激励函数都在这
import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import train_test_split
import torch.nn as nn
import shap
import seaborn as sns
# import xgboost

plt.close('all')
# csv_data = pd.read_csv('D:/experiments/ML project of Invar alloys/Code for ANN/Data base.xlsx')  # 读取训练数据
# print(csv_data.shape)  # (189, 9)
data = pd.read_excel(r"D:\experiments\ML project of Invar alloys\Code for ANN\data\Data_base.xlsx")
df_test = data.drop(columns=['alloy'])

def minmaxscaler(data):
    min = np.amin(data)
    max = np.amax(data)    
    return (data - min)/(max-min)


# x1 = df_test [['Fe','Ni','Co','Cr','V','Cu']]
# x1 = df_test [['Fe','Ni','Co','Cr','V','Cu','VEC']]
# x1 = df_test [['Fe','Ni','Co','Cr','V','Cu','NM']]
# x1 = df_test [['Fe','Ni','Co','Cr','V','Cu']]
x1 = df_test [['Fe','Ni','Co','Cr','V','Cu']]


x2 = df_test [['Density','VEC','AR1','AR2','PE','TC','MP','FI','SI','TI','M']]

# x2 = df_test [['NM']]

x2_normalization = minmaxscaler(x2)

x_combined = np.concatenate ((x1, x2_normalization), axis=1)

y1 = df_test [['TEC']]


x = torch.FloatTensor(x_combined) 
# x = torch.FloatTensor(x1.values) 
y = torch.FloatTensor(y1.values)
  # noisy y data (tensor), shape=(100, 1)

# print (x.shape)

# named_x = x.refine_names('Fe', 'Ni')

# print(x,named_x)

train_features, test_features, train_labels, test_labels = train_test_split(x, y, test_size=0.15, random_state=40)


batch_size = 16

print(train_features)
# print(x)
# print(y)
# plt.plot(x , y, "ro")
# plt.show()




class Net(nn.Module):  # 继承 torch 的 Module
    def __init__(self, n_feature, n_hidden, n_output):
        super(Net, self).__init__()     # 继承 __init__ 功能
        # 定义每层用什么样的形式
        self.hidden1 = torch.nn.Linear(n_feature, n_hidden) 
        self.hidden2 = torch.nn.Linear(n_hidden, n_hidden)# 隐藏层线性输出
        self.hidden3 = torch.nn.Linear(n_hidden, n_hidden)# 隐藏层线性输出
        self.hidden4 = torch.nn.Linear(n_hidden, n_hidden)# 隐藏层线性输出
        self.predict = torch.nn.Linear(n_hidden, n_output)   # 输出层线性输出

    def forward(self, x):   # 这同时也是 Module 中的 forward 功能
        # 正向传播输入值, 神经网络分析出输出值
        x = self.hidden1(x)
        x = F.relu(x)      # 激励函数(隐藏层的线性值)
        
        x = self.hidden2(x)
        x = F.relu(x) 
               
        x = self.hidden3(x)
        x = F.relu(x) 
        
        x = self.hidden4(x)
        x = F.relu(x) 

        
        x = self.predict(x)             # 输出值
        return x

net = Net(n_feature=17, n_hidden=16, n_output=1)

print(net)  # net 的结构】

class MAPELoss(nn.Module) :
     def __init__(self):
        super(MAPELoss, self).__init__() 
        
     def forward (self, output, target):
         loss = torch.mean(torch.abs((target - output) / target))
         return loss

optimizer = torch.optim.Adam(net.parameters(), lr=0.0005)  # 传入 net 的所有参数, 学习率
loss = MAPELoss()      # 预测值和真实值的误差计算公式 (均方差)
# loss = MAPELoss() 
num_epochs=10000



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
            optimizer.zero_grad()
            ls.backward()
            optimizer.step()
        
        # save loss for each step
        train_ls.append(loss(net(train_features).view(-1, 1), train_labels.view(-1, 1)).item())
        test_ls.append(loss(net(test_features).view(-1, 1), test_labels.view(-1, 1)).item())
        
        if (epoch % 100 == 0):
            print ("epoch %d: train loss %f, test loss %f" % (epoch, train_ls[-1], test_ls[-1]))
    
    print ("final epoch: train loss %f, test loss %f" % (train_ls[-1], test_ls[-1]))
    
    print ("plot curves")
    plotCurve(range(1, num_epochs + 1), train_ls,
              "epoch", "loss",
              range(1, num_epochs + 1), test_ls,
              ["train", "test"]
              )
    plt.text(6000, 0.8, 'Loss=%.4f' % test_ls[-1], fontdict={'size': 20, 'color':  'red'})
    plt.savefig('D:\\experiments\\ML project of Invar alloys\\Code for ANN\\results\\3_16BS_17elements_4_16_hidden_MAPEloss_Adam_lr0.0005_epoch_10000\\5_1.png', format='png', dpi=300)
    print ("=== train end ===")

    
train(net, num_epochs, batch_size, train_features, train_labels, test_features, test_labels, loss, optimizer)

# for i in range(10000):

    
#     inputs=x_train
    
#     target=y_train
    
#     out=net(inputs)
    
#     loss=criterion(out,target)
    
#     optimizer.zero_grad()
#     loss.backward()
#     optimizer.step()
    

    
#     # for p in net.parameters():
       
#     #     print(p.data, p.grad)
    
#     # f = loss.grad_fn
#     # while True:
#     #      print(f)
#     #      if len(f.next_functions) == 0:
#     #           break
#     #      f = f.next_functions[0][0]

net.eval()
predict=net(test_features)
predict=predict.data.numpy()

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
sns.regplot(x=test_labels.numpy().reshape(-1,1), y=predict, color='g')  
plt.savefig('D:\\experiments\\ML project of Invar alloys\\Code for ANN\\results\\3_16BS_17elements_4_16_hidden_MAPEloss_Adam_lr0.0005_epoch_10000\\5_2.png', format='png', dpi=300)




explainer = shap.DeepExplainer(net,train_features)
shap_values = explainer.shap_values(train_features)

matplotlib=True 
# # visualize the first prediction's explanation (use matplotlib=True to avoid Javascript)


shap.dependence_plot("Feature 2",shap_values, train_features.numpy())
plt.xlabel('Co')
plt.ylabel('SHAP value for Co')
plt.savefig('D:\\experiments\\ML project of Invar alloys\\Code for ANN\\results\\3_16BS_17elements_4_16_hidden_MAPEloss_Adam_lr0.0005_epoch_10000\\5_3.png', format='png', dpi=300)

shap.dependence_plot("Feature 0",shap_values, train_features.numpy())
plt.xlabel('Fe')
plt.ylabel('SHAP value for Fe')
plt.savefig('D:\\experiments\\ML project of Invar alloys\\Code for ANN\\results\\3_16BS_17elements_4_16_hidden_MAPEloss_Adam_lr0.0005_epoch_10000\\5_4.png', format='png', dpi=300)

shap.dependence_plot("Feature 1",shap_values, train_features.numpy())
plt.xlabel('Ni')
plt.ylabel('SHAP value for Ni')
plt.savefig('D:\\experiments\\ML project of Invar alloys\\Code for ANN\\results\\3_16BS_17elements_4_16_hidden_MAPEloss_Adam_lr0.0005_epoch_10000\\5_5.png', format='png', dpi=300)

shap.dependence_plot("Feature 4",shap_values, train_features.numpy())
plt.xlabel('V')
plt.ylabel('SHAP value for V')
plt.savefig('D:\\experiments\\ML project of Invar alloys\\Code for ANN\\results\\3_16BS_17elements_4_16_hidden_MAPEloss_Adam_lr0.0005_epoch_10000\\5_6.png', format='png', dpi=300)

shap.dependence_plot("Feature 5",shap_values, train_features.numpy())
plt.xlabel('Cu')
plt.ylabel('SHAP value for Cu')
plt.savefig('D:\\experiments\\ML project of Invar alloys\\Code for ANN\\results\\3_16BS_17elements_4_16_hidden_MAPEloss_Adam_lr0.0005_epoch_10000\\5_7.png', format='png', dpi=300)

shap.dependence_plot("Feature 3",shap_values, train_features.numpy())
plt.xlabel('Cr')
plt.ylabel('SHAP value for Cr')
plt.savefig('D:\\experiments\\ML project of Invar alloys\\Code for ANN\\results\\3_16BS_17elements_4_16_hidden_MAPEloss_Adam_lr0.0005_epoch_10000\\5_8.png', format='png', dpi=300)

plt.figure()
shap.summary_plot(shap_values, train_features, max_display=30)
# plt.yticks([0, 1, 2, 3, 4, 5],['$Cu$', '$V$', '$Cr$', '$Fe$', '$Co$', '$Ni$'])
plt.savefig('D:\\experiments\\ML project of Invar alloys\\Code for ANN\\results\\3_16BS_17elements_4_16_hidden_MAPEloss_Adam_lr0.0005_epoch_10000\\5_9.png', format='png', dpi=300)


plt.figure()
shap.summary_plot(shap_values, train_features, plot_type="bar")
# plt.yticks([0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16],['$Bohr magneton$', '$Cu$', '$Cr$', '$Fe$', '$Co$', '$Ni$','$Cu$', '$V$', '$Cr$', '$Fe$', '$Co$', '$Ni$'])
plt.savefig('D:\\experiments\\ML project of Invar alloys\\Code for ANN\\results\\3_16BS_17elements_4_16_hidden_MAPEloss_Adam_lr0.0005_epoch_10000\\5_10.png', format='png', dpi=300)
# print(shap_values)
# plt.figure()
# j = 30
# shap.initjs()
# train_features=pd.DataFrame(train_features)
# print(train_features)
# shap.force_plot(explainer.expected_value, shap_values[j], train_features.iloc[j])



plt.show()

torch.save(net, 'D:\\experiments\\ML project of Invar alloys\\Code for ANN\\results\\3_16BS_17elements_4_16_hidden_MAPEloss_Adam_lr0.0005_epoch_10000\\5_net.pkl')  # 保存整个网络
torch.save(net.state_dict(), 'D:\\experiments\\ML project of Invar alloys\\Code for ANN\\results\\3_16BS_17elements_4_16_hidden_MAPEloss_Adam_lr0.0005_epoch_10000\\5_net_params.pkl')   # 只保存网络中的参数 (速度快, 占内存少)

# net2 = torch.load('D:\\experiments\\ML project of Invar alloys\\Code for ANN\\results\\3_16BS_17elements_4_16_hidden_MAPEloss_Adam_lr0.0005_epoch_10000\\1_net.pkl')
# predict1 = net2(test_features)
# print(predict, predict1.data.numpy())

# 将保存的参数复制到 net
# net.load_state_dict(torch.load('D:\\experiments\\ML project of Invar alloys\\Code for ANN\\results\\3_16BS_17elements_4_16_hidden_MAPEloss_Adam_lr0.0005_epoch_10000\\1_net_params.pkl'))
# predict2 = net(x)
# print(predict2.data.numpy())