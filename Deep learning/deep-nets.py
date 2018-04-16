# -*- coding: utf-8 -*-
"""
Created on Sun Apr 15 21:13:20 2018

@author: Ray Liu
"""

import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

"""
input > weight >hidden layer 1 (activation function) > weights > hidden l2(act func)
> weights > output layer

compare output to intended output -> costfunction  (cross entropy)
optimization function (optimizer) -> minimize cost (AdamOptimizer)

backpropagation ^^


feed forward + backprop = epoch (cycle)

"""

mnist = input_data.read_data_sets("/tmp/data/", one_hot= True)

#10 classes, 0 - 9

'''
What one_hot does
eg
0 = [1,0,0,0,0,0,0,0,0,0]
'''

n_nodes_hl1 = 500
n_nodes_hl2 = 500
n_nodes_hl3 = 500

n_classes = 10
batch_size = 100

x = tf.placeholder('float',[None, 28*28])
y = tf.placeholder('float' )

def neural_network_model(data):
    
    # input_data * weights + biases
    hidden_1_layer = {'weights': tf.Variable(tf.random_normal([784,n_nodes_hl1])),
                      'biases': tf.Varaibles(tf.random_normal(n_nodes_hl1))}
    
    hidden_2_layer = {'weights': tf.Variable(tf.random_normal([nodes_hl1,n_nodes_hl2])),
                      'biases': tf.Varaibles(tf.random_normal(n_nodes_hl2))}
    hidden_3_layer = {'weights': tf.Variable(tf.random_normal([n_nodes_hl2,n_nodes_hl3])),
                      'biases': tf.Varaibles(tf.random_normal(n_nodes_hl3))}
    output_layer = {'weights': tf.Variable(tf.random_normal([n_nodes_hl3,n_classes])),
                      'biases': tf.Varaibles(tf.random_normal(n_classes))}
    #(data*weight) + biases
    l1 = tf.add(tf.matmul(data,hidden_1_layer['weights'])+hidden_1_layer['biases'])
    #activation function
    l1 = tf.nn.relu(l1)
    
    #(data*weight) + biases
    l2 = tf.add(tf.matmul(data,hidden_2_layer['weights'])+hidden_2_layer['biases'])
    #activation function
    l2 = tf.nn.relu(l2)
    
    #(data*weight) + biases
    l3 = tf.add(tf.matmul(data,hidden_3_layer['weights'])+hidden_3_layer['biases'])
    #activation function
    l3 = tf.nn.relu(l3)
    
    output = tf.matmul(data,output_layer['weights'])+output_layer['biases']
    
    return output