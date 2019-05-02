
"""
Created on Thu Apr 25 12:14:35 2019

@author: xiaoc
"""

import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt 


def add_layer(inputs,in_size,out_size,n_layer,activation_function=None): #add more layers and return th out put of this layer
    layer_name = 'layer%s'%n_layer
    with tf.name_scope('layer_name'):
        with tf.name_scope('weight'):
            Weights = tf.Variable(tf.random_normal([in_size,out_size]),name = 'W') #一个in行 out列的矩阵
            tf.summary.histogram(layer_name + 'weight', Weights)
        with tf.name_scope('biases'):
            biases = tf.Variable(tf.zeros([1,out_size])+0.1,name='b')
            tf.summary.histogram(layer_name + 'biases', biases)
        with tf.name_scope('Wx_plus_b'):
            Wx_plus_b =tf.matmul(inputs,Weights) + biases
#    1.tf.multiply（）两个矩阵中对应元素各自相乘     tf.matmul（）将矩阵a乘以矩阵b，生成a * b
        if activation_function is None:
            outputs = Wx_plus_b
        else:
            outputs = activation_function(Wx_plus_b)
        tf.summary.histogram(layer_name + 'output',outputs)
        return outputs

x_data = np.linspace(-1,1,300) [:,np.newaxis]
 #-1 到1 区间的 300个点  后面是增加一个维度。
 #这样改变维度的作用往往是将一维的数据转变成一个矩阵，与代码后面的权重矩阵进行相乘， 否则单单的数据是不能呢这样相乘的哦。
noise = np.random.normal(0,0.05,x_data.shape)

y_data = np.square(x_data) - 0.5 + noise

with tf.name_scope('inputs'):
    
    xs = tf.placeholder(tf.float32,[None,1],name = 'x_input')
    ys = tf.placeholder(tf.float32,[None,1],name = 'y_input')  # 多行 1 列的数组

l1 = add_layer(xs,1,10,n_layer = 1,activation_function=tf.nn.relu)
prediction = add_layer(l1,10,1,n_layer = 1,activation_function=None)

with tf.name_scope('loss'):
    loss = tf.reduce_mean(tf.reduce_sum(tf.square(ys-prediction),reduction_indices = [1]))
    tf.summary.scalar('loss',loss)
with tf.name_scope('train'):
    train_step = tf.train.GradientDescentOptimizer(0.1).minimize(loss)
#tf.reduce_mean axis=0 做最外围计算  axis越大，往维度越小计算   axis最大就是 最小单位的计算
init = tf.global_variables_initializer()

sess = tf.Session()

writer = tf.summary.FileWriter("log/",sess.graph)
writer.add_graph(sess.graph)
#fig = plt.figure()
#ax = fig.add_subplot(1,1,1)
#ax.scatter(x_data,y_data)
#plt.ion()
#plt.show()
sess.run(init)
merged = tf.summary.merge_all()
for i in range(1000):
    sess.run(train_step,feed_dict={xs:x_data,ys:y_data})
    if i%50 == 0:
#        print(sess.run(loss,feed_dict={xs:x_data,ys:y_data}))
        result = sess.run(merged,feed_dict={xs:x_data,ys:y_data})
        writer.add_summary(result,i)
#        try:
#            ax.lines.remove(lines[0])
#        except Exception:
#            pass
        prediction_value = sess.run(prediction, feed_dict={xs:x_data})
        lines = xs.plot(x_data,prediction_value,'r-',lw=5)
        plt.pause(0.2)


