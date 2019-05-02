# -*- coding: utf-8 -*-
"""
Created on Mon Apr 29 09:54:16 2019

@author: xiaoc
"""

import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

mnist = input_data.read_data_sets('MNIST_data',one_hot=True)

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

def compute_accuracy(v_xs,v_ys):
    global prediction
    y_pre = sess.run(prediction,feed_dict={xs:v_xs})
    correct_prediction = tf.equal(tf.argmax(y_pre,1),tf.argmax(v_ys,1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction,tf.float32))
    result = sess.run(accuracy,feed_dict={xs:v_xs,ys:v_ys})
    return result

#define placeholder for inputs tp network 
xs = tf.placeholder(tf.float32,[None,784]) #28*28
ys = tf.placeholder(tf.float32,[None,10])

#app outlayer 
prediction = add_layer(xs,784,10,activation_function=tf.nn.softmax)

#error between real and prediction 

cross_entropy = tf.reduce_mean(-tf.reduce_sum(ys*tf.log(prediction),reduction_indices=[1])) 
train_step = tf.train.GradientDescentOptimizer(0.5).minimize(cross_entropy)
sess= tf.Session()
sess.run(tf.global_variables_initializer())
for i in range(10000):
    batch_xs,batch_ys = mnist.train.next_batch(100)
    sess.run(train_step,feed_dict={xs:batch_xs,ys:batch_ys})
    if i%50 == 0:
        print(compute_accuracy(mnist.test.images,mnist.test.labels))