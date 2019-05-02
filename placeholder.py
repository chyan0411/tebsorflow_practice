# -*- coding: utf-8 -*-
"""
Created on Thu Apr 25 12:07:22 2019

@author: xiaoc
"""

import tensorflow as tf
input1 = tf.placeholder(tf.float32)
input2 = tf.placeholder(tf.float32)
#placeholder feed_dict 成对出现
output = tf.multiply(input1,input2)
with tf.Session() as sess:
     print(sess.run(output,feed_dict = {input1:[7.],input2:[2.]}))
     
