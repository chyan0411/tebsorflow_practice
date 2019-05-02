# -*- coding: utf-8 -*-
"""
Created on Thu Apr 25 12:14:35 2019

@author: xiaoc
"""

import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt 


def add_layer(inputs,in_size,out_size,activation_function=None): #add more layers and return th out put of this layer
    Weights = tf.Variable(tf.random_normal([in_size,out_size])) #一个in行 out列的矩阵
    biases = tf.Variable(tf.zeros([1,out_size])+0.1)
    Wx_plus_b =tf.matmul(inputs,Weights) + biases
#    1.tf.multiply（）两个矩阵中对应元素各自相乘     tf.matmul（）将矩阵a乘以矩阵b，生成a * b
    if activation_function is None:
        outputs = Wx_plus_b
    else:
        outputs = activation_function(Wx_plus_b)
    return outputs

x_data = np.linspace(-1,1,300) [:,np.newaxis]
 #-1 到1 区间的 300个点  后面是增加一个维度。
 #这样改变维度的作用往往是将一维的数据转变成一个矩阵，与代码后面的权重矩阵进行相乘， 否则单单的数据是不能呢这样相乘的哦。
noise = np.random.normal(0,0.05,x_data.shape)

y_data = np.square(x_data) - 0.5 + noise

xs = tf.placeholder(tf.float32,[None,1])
ys = tf.placeholder(tf.float32,[None,1])  # 多行 1 列的数组

l1 = add_layer(xs,1,10,activation_function=tf.nn.relu)
prediction = add_layer(l1,10,1,activation_function=None)


loss = tf.reduce_mean(tf.reduce_sum(tf.square(ys-prediction),reduction_indices = [1]))
train_step = tf.train.GradientDescentOptimizer(0.1).minimize(loss)
#tf.reduce_mean axis=0 做最外围计算  axis越大，往维度越小计算   axis最大就是 最小单位的计算
init = tf.initialize_all_variables()

sess = tf.Session()
sess.run(init)

fig = plt.figure()
ax = fig.add_subplot(1,1,1)
ax.scatter(x_data,y_data)
plt.ion()
plt.show()

for i in range(1000):
    sess.run(train_step,feed_dict={xs:x_data,ys:y_data})
    if i%50 == 0:
#        print(sess.run(loss,feed_dict={xs:x_data,ys:y_data}))
        try:
            ax.lines.remove(lines[0])
        except Exception:
            pass
        prediction_value = sess.run(prediction, feed_dict={xs:x_data})
        lines = ax.plot(x_data,prediction_value,'r-',lw=5)
        plt.pause(0.2)


