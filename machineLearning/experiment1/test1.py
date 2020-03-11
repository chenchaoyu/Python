import torch
import random
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.datasets import load_svmlight_file
x_train, y_train = load_svmlight_file("a9a.txt")
x_test,y_test=load_svmlight_file("a9at.txt")
x_train=x_train.toarray()
x_test=x_test.toarray()



b=np.ones(len(x_train))
x_train=np.c_[b,x_train]
b=np.ones(len(x_test))
x_test=np.c_[b,x_test]
b=np.zeros(len(x_test))
x_test=np.c_[x_test,b]
w=np.random.randn(124,1)

for i in range(len(y_train)):
    if y_train[i]==-1: 
        y_train[i]=0

for i in range(len(y_test)):
    if y_test[i]==-1:
        y_test[i]=0

#sigmoid函数
def sigmoid(inX):
    return 1.0 / (1 + np.exp(-inX))

def loss(x,y,w):
    return np.sum(y*np.dot(x,w)-np.log(1+np.exp(np.dot(x,w))))

# 求导
def gradient(x,y,w):
    return np.dot(x.T,sigmoid(np.dot(x,w))-y)/len(x)
#随机取样本
def rand(batch,X,Y):
    m=X.shape[0]
    n=X.shape[1]
    sam_ids=[i for i in sorted(random.sample(range(m),batch))]
    x=np.zeros((batch,n))
    for i in range(batch):
        index=sam_ids[i]
        x[i]=X[index]
    y=np.zeros((batch,1))
    for i in range(batch):
        index=sam_ids[i]
        y[i]=Y[index]
    return x,y
#    求准确率
def right(y,z):
        count=0
        for i in range(len(y)):
            if y[i][0]==z[i]:
                count+=1
                
        return 1.0*count/len(y)
#求1的次数
def time(x):
    count=0
    for i in range(len(x)):
        if x[i]==1:
            count+=1
    return count

for i in range(1000):
    x1,y1=rand(10,x_train,y_train)
    w=w-0.01*gradient(x1,y1,w)
    J=loss(x_train,y_train,w)
    print(J)
y_pred=np.zeros([len(x_test),1])
#取得预测值，阈值为0.5
for i in range(len(x_test)):
    if sigmoid(np.dot(x_test[i],w)[0])>0.5:
        y_pred[i]=1
    else:
        y_pred[i]=0
        # Lvalidation=loss(x_test,y_pred)
        # print(Lvalidation)
print('准确率:',right(y_pred,y_test))









