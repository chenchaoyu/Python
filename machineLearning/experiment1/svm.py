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
    x,y_train=ds.load_svmlight_file("./a9a.txt")
    x_train=x.toarray()
   
    x_,y_test=ds.load_svmlight_file("./a9at.txt")
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


def dJ(w,x,y,c):
    n=x.shape[1]
    dJ=np.zeros((n,1))
    for i in range(x.shape[0]):
        judge=1-y[i]*np.dot(x[i],w)
        dW=np.zeros((n,1))
        if(judge[0]>=0):
            dW=-1*y[i]*x[i]
            dJ=dJ+dW.reshape((n,1))
      
        
    
    dJ=c*dJ/n
    dJ=w+dJ
   
    return dJ

def svm():
    batch_size=100
    W=np.zeros([124,1])
    i=0

    while i<1000:
        x,y=next_batch(batch_size,x_train,y_train)
        dJ_=dJ(W,x,y,10)
        W=W-0.1*dJ_
        i+=1
    #print(W)
    #print(dJ(W,x,y,1).shape)
    evaluate(W)
  


def ThetaX(w,x):
    return np.dot(x,w)

def classification(x):
    
    y=np.zeros((x.shape[0],1))
    for i in range(x.shape[0]):
        if x[i][0]<-1:
            y[i][0]=-1
        else:
             y[i][0]=1
    return y
      
def J(w,x,y):
      y_hat  = ThetaX(w,x)
      try:
        return np.sum(y*np.log(y_hat)+(1-y)*np.log(1-y_hat)) / len(y)
      except:
        return float('inf')

def evaluate(W):
    
    x,y=next_batch(500,x_test,y_test)
    m=y.shape[0]
    y_predit=ThetaX(W,x)
    y_=classification(y_predit)
    ct=0
    cty=0
    cty_=0

    for i in range(m):
        print(y_[i],end=" ")
        print(y[i])
    
   

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

def test():
    dJ=np.zeros((n,1))
load_data()
svm()