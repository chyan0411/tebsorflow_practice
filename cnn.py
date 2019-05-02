# -*- coding: utf-8 -*-
"""
Created on Mon Apr 29 09:54:16 2019

@author: xiaoc
"""

import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

mnist = input_data.read_data_sets('MNIST_data',one_hot=True)

#def add_layer(inputs,in_size,out_size,activation_function=None): #add more layers and return th out put of this layer
#    Weights = tf.Variable(tf.random_normal([in_size,out_size])) #一个in行 out列的矩阵
#    biases = tf.Variable(tf.zeros([1,out_size])+0.1)
#    Wx_plus_b =tf.matmul(inputs,Weights) + biases
##    1.tf.multiply（）两个矩阵中对应元素各自相乘     tf.matmul（）将矩阵a乘以矩阵b，生成a * b
#    if activation_function is None:
#        outputs = Wx_plus_b
#    else:
#        outputs = activation_function(Wx_plus_b)
#    return outputs

def compute_accuracy(v_xs,v_ys):
    global prediction
    y_pre = sess.run(prediction,feed_dict={xs:v_xs,keep_prob:1})
    correct_prediction = tf.equal(tf.argmax(y_pre,1),tf.argmax(v_ys,1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction,tf.float32))
    result = sess.run(accuracy,feed_dict={xs:v_xs,ys:v_ys,keep_prob:1})
    return result

def weight_variable(shape):
    initial = tf.truncated_normal(shape, stddev=0.1)
    return tf.Variable(initial)
def bias_variable(shape):
    initial = tf.constant(0.1, shape =shape)
    return tf.Variable(initial)
def conv2d(x,W):
#    strides = [1,x move,y move,1]
#    padding  valid  and same :must have strides[0]=[3]=1
    return tf.nn.conv2d(x,W,strides=[1,1,1,1],padding = 'SAME')
def max_pooling(x):
    return tf.nn.max_pool(x,ksize=[1,2,2,1], strides =[1,2,2,1],padding = 'SAME')



#define placeholder for inputs tp network 
xs = tf.placeholder(tf.float32,[None,784]) #28*28
ys = tf.placeholder(tf.float32,[None,10])
keep_prob = tf.placeholder(tf.float32)
x_image = tf.reshape(xs,[-1,28,28,1])
# print(x_image.shape) #[n_sample,28,28,1] 

#conv1 layer 
W_conv1 = weight_variable([5,5,1,32])  #5*5  input 1 ,output32 
b_conv1 = bias_variable([32])
h_conv1 = tf.nn.relu(conv2d(x_image,W_conv1)+b_conv1) #output 28*28*32
h_pool1 = max_pooling(h_conv1) # output 14*14*32

#conv2 layer
W_conv2 = weight_variable([5,5,32,64])  #5*5  input32 ,output64 
b_conv2 = bias_variable([64])
h_conv2 = tf.nn.relu(conv2d(h_pool1,W_conv2)+b_conv2) #output 14*14*64
h_pool2 = max_pooling(h_conv2) # output 7*7*64

#func1 layer
W_func1 = weight_variable([7*7*64,1024])
b_func1 = bias_variable([1024]) 
#[n_samples,7,7,64] ->> [n_samples,7*7*64]
h_pool2_flat = tf.reshape(h_pool2,[-1,7*7*64])
h_func1 = tf.nn.relu(tf.matmul(h_pool2_flat,W_func1)+b_func1)
h_func1_drop = tf.nn.dropout(h_func1 ,keep_prob)

#func2 layer
W_func2 = weight_variable([1024,10])
b_func2 = bias_variable([10]) 
prediction = tf.nn.softmax(tf.matmul(h_func1_drop,W_func2)+b_func2)


#error between real and prediction 

cross_entropy = tf.reduce_mean(-tf.reduce_sum(ys*tf.log(prediction),reduction_indices=[1])) 
train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)
sess= tf.Session()
sess.run(tf.global_variables_initializer())
for i in range(1000):
    batch_xs,batch_ys = mnist.train.next_batch(100)
    sess.run(train_step,feed_dict={xs:batch_xs,ys:batch_ys,keep_prob:0.5})
    if i%50 == 0:
        print(compute_accuracy(mnist.test.images[:1000],mnist.test.labels[:1000]))