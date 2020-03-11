import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

#定义隐藏层
def add_layer(input,in_size,out_size,activation_function=None):
    Weights=tf.Variable(tf.random_normal([in_size,out_size]))#权值
    biases=tf.Variable(tf.zeros([1,out_size])+0.1)
    Z=tf.matmul(input,Weights)+biases
    if activation_function is None:
        outputs=Z
    else:
        outputs=activation_function(Z)
    return outputs

#制作输入数据
x_data=np.linspace(-1,1,300)[:,np.newaxis]
noise=np.random.normal(0,0.05,x_data.shape)
y_data=np.square(x_data)-0.5+noise

#构建图
xs=tf.placeholder(tf.float32,[None,1])
ys=tf.placeholder(tf.float32,[None,1])

#建立第一第二隐藏层
l1=add_layer(xs,1,10,activation_function=tf.nn.relu)
prediction=add_layer(l1,10,1,activation_function=None)

#创建损失函数
loss=tf.reduce_mean(tf.reduce_sum(tf.square(ys-prediction),
                    reduction_indices=[1])
                    )
train_step=tf.train.GradientDescentOptimizer(0.1).minimize(loss)

#important step
init=tf.initialize_all_variables()
sess=tf.Session()#创建会话框
sess.run(init)

#绘图部分
# fig=plt.plot(x_data,y_data)
ax=plt.scatter(x_data,y_data)

plt.show()



#学习1000步
for i in range(1000):
    sess.run(train_step,feed_dict={xs:x_data,ys:y_data})
    if i%50==0:
        try:
            
+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++        plt.pause(0.1)
      
            



