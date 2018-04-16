# -*- coding: utf-8 -*-
"""
Created on Sun Apr 15 20:36:46 2018

@author: Ray Liu
"""

import tensorflow as tf

x1 = tf.constant(5)
x2 = tf.constant(6)

result = tf.multiply(x1,x2)

print(result)

with tf.Session() as sess:
    print(sess.run(result))
    