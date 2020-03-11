from sklearn import datasets as ds
from sklearn.model_selection import train_test_split as data_split
from random import randint
from sklearn.linear_model import LinearRegression
import numpy as np
from matplotlib.pyplot import *



#导入处理数据
def load_and_process_data():
    #导入数据
    x,Y=ds.load_svmlight_file("./housing_scale.txt",n_features=13)

    #将x从csr_matrix转为array
    X=x.toarray()

    #切分数据
    global x_train,x_test,y_train,y_test
    x_train,x_test,y_train,y_test=data_split(X,Y,test_size=0.3,random_state=1)

    #给X增加一列
    b=np.ones(len(x_train))
    x_train=np.c_[b,x_train]
    b=np.ones(len(x_test))
    x_test=np.c_[b,x_test]

def return_loss(w,x,y):
    y_predit=np.matmul(x,w).flatten()
    sq=np.square(y_predit-y)
    loss=np.sum(sq,0)/(2*len(y))
    return loss


def linear_regression_normal_equation():
    #生成参数(服从正态分布)
    W=np.random.randn(14,1)

    print("normal_equation start*******************************")
   

    print("solving W......")
    #闭式解求W ((XT*X)-1)*XT*Y

    #XT*X
    temp1=np.matmul(x_train.T,x_train)

    #求逆((XT*X)-1)
    inv=np.linalg.inv(temp1)

    #((XT*X)-1)*XT
    temp2=np.matmul(inv,x_train.T)

    #((XT*X)-1)*XT*Y
    W=np.matmul(temp2,y_train)
    W.reshape([14,1])

    print("W:")
    print(W)

    print("loss_train:")
    print(return_loss(W,x_train,y_train))
    print("loss_val:")
    print(return_loss(W,x_test,y_test))
    print("normal_equation end*******************************")

    y_test_predit=np.matmul(x_test,W).flatten()

    scatter(y_test_predit,y_test,s=75,alpha=1)

    show()

def return_dJ(w_now,data_x,y_true):
    #预估的y
    y_estimate=np.matmul(data_x,w_now)
    y_estimate=y_estimate.flatten()
    
    #特征数
    feature_num=data_x.shape[0]
    #构建dJ
    _dJ=np.zeros([feature_num,1])

    for i in range(feature_num):
        _dJ[i,0]=((y_estimate[0]-y_true)*data_x[i])
        
    return _dJ



def linear_regression_random_gradient_descent():

    print("random_gradient_descent start*******************************")
     #生成参数(服从正态分布)
    W=np.random.randn(14,1)


    #梯度下降求W
    learning_rate=0.01
    ct=0
    ER=10
    flag=0
    print("solving W......")
    while flag==0 and ct<10000:
        
        last_W=W
        id=randint(0,x_train.shape[0]-1)
        gradient=return_dJ(W,x_train[id],y_train[id])
        W=W-learning_rate*gradient
        loss=return_loss(W,x_train,y_train)
         
        if abs(loss)<ER:
            flag=1
        ct+=1
    W.reshape([14,1])
    print(W)
    print("loss_train:")
    print(return_loss(W,x_train,y_train))
    print("loss_val:")
    print(return_loss(W,x_test,y_test))
    print("random_gradient_descent end*******************************")
    y_test_predit=np.matmul(x_test,W).flatten()

    scatter(y_test_predit,y_test,s=75,alpha=1)

    show()

load_and_process_data()

linear_regression_normal_equation()
linear_regression_random_gradient_descent()









