import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl


alpha=0.08
r=20.0
Wmax=15000.0
W0=30

#Defintion of ODE
def get_w_dot(w):
    return r*w*(alpha-w/Wmax)

#Solution to get w
def w(t):
    w=W0
    delta_t=0.001
    for time in np.arange(0,t,delta_t):
        w_dot=get_w_dot(w)
        w+=w_dot*delta_t
    return w

tarray=np.arange(0,15,0.01)
warray=[]
print(tarray)

for t in tarray:
    temp=w(t)
    warray=np.append(warray,temp)
print(warray)
plt.title('weight-time')
plt.xlabel('w(t)/kg')
plt.ylabel('time/year')

plt.plot(tarray,warray,'k', linewidth=2,label='lgx')
plt.show()

    
