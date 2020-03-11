from sklearn import datasets as ds
from sklearn.model_selection import train_test_split as data_split
from random import randint
from random import sample
import math
from sklearn.linear_model import LinearRegression
import numpy as np
from matplotlib.pyplot import *

def load_data():
    global x_train,x_test,y_train,y_test
    x,y_train=ds.load_svmlight_file("./b9b.txt")
    x_train=x.toarray()
   
    x_,y_test=ds.load_svmlight_file("./b9bt.txt")
    x_test=x_.toarray()
  
    
   

    b=np.ones(len(x_train))
    x_train=np.c_[b,x_train]

    b=np.ones(len(x_test))
    x_test=np.c_[b,x_test]
    b=np.zeros(len(x_test))
    x_test=np.c_[x_test,b]
    

def next_batch(batch_size,X,Y):
    m=X.shape[0]
    n=X.shape[1]

    sam_ids=[i for i in sorted(sample(range(m),batch_size))]
    x=np.zeros((batch_size,n))
    for i in range(batch_size):
        index=sam_ids[i]
        x[i]=X[index]
    y=np.zeros((batch_size,1))
    for i in range(batch_size):
        index=sam_ids[i]
        y[i]=Y[index]
    return x,y
  
def sigmod(x):
    return 1/(1+np.exp(-x))

def sigmodThetaX(w,x):
    return sigmod(np.dot(x,w))

def dJ(w,x,y):
    return np.dot(x.T,sigmodThetaX(w,x)-y)/len(x)
    #m=x.shape[0]
    #A=sigmodThetaX(w,x)
    #n=x.shape[1]
    #dJ=np.zeros((n,1))
    #for i in range(n):
        #dJ[i]=np.dot(x[:,i],(A-y))
    #return dJ
    
    

    
        
def classification(x):
    
    y=np.zeros(x.shape)
    for i in range(x.shape[0]):
        if x[i][0]<0.5:
            y[i][0]=0
        else:
             y[i][0]=1
    return y
      

def logistic_regression():
    
    batch_size=100
    flag=0
    W=np.zeros((124,1))
    i=0
    
    while i<10000 and flag==0:
        x,y=next_batch(batch_size,x_train,y_train)
        dJ_=dJ(W,x,y)
        W=W-0.01*dJ_
        i+=1
    
    #x,y=next_batch(1000,x_train,y_train)
    '''for i in range(1000):
        j=randint(1,900)
        h=sigmod(np.sum(x[j]*W))
        error=y[j]-h
        
        W=W+0.01*error*(x[j].reshape(124,1))
    '''
    evaluate(W)
    print(W)
    
    
def J(w,x,y):
      y_hat  = sigmodThetaX(w,x)
      try:
        return np.sum(y*np.log(y_hat)+(1-y)*np.log(1-y_hat)) / len(y)
      except:
        return float('inf')
      
 
    
def evaluate(W):
    
    x,y=next_batch(500,x_test,y_test)
    m=y.shape[0]
    y_predit=sigmodThetaX(W,x)
    y_=classification(y_predit)
    ct=0
    cty=0
    cty_=0

    for i in range(m):
        print(y_[i],end=" ")
        print(y[i])
    
    loss=J(W,x,y)
    print("loss",end=" ")
    print(loss)

    for i in range(m):
        if y[i]>0:
            cty+=1
    postive_rate=cty/m
    print("postive_rate_for_y:",end=" ")
    print(postive_rate)

    for i in range(m):
        if y_[i]>0:
            cty_+=1
    postive_rate=cty_/m
    print("postive_rate_for_y_:",end=" ")
    print(postive_rate)

    for i in range(m):
        if y_[i]==y[i]:
            ct+=1
    accuracy=ct/m
    print("accuracy:",end=" ")
    print(accuracy)


def get_H(w,x,y):
    temp1=np.log(theta(w,x))
    
    temp2=temp1*y
    ones=np.ones((y.shape[0],1))
    temp3=(ones-y)
    temp4=np.log(ones-h_x(w,x))
    temp5=temp3*temp4
    return np.sum(temp2+temp5)
   
    
    

def test():

        x,y=next_batch(500,x_train,y_train)
        '''
        for i in range(100):
            print(y[i])
        cty=0
        for i in range(100):
            if y[i]>0:
                cty+=1
        postive_rate=cty/100
        print("postive_rate:",end=" ")
        print(postive_rate)
        '''
        
    

load_data()
#test()
logistic_regression()

