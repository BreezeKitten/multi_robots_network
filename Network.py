# -*- coding: utf-8 -*-
"""
Created on Tue Jun 16 15:18:17 2020

@author: BreezeCat
"""

import tensorflow.compat.v1 as tf


number_of_state = 15 #for relative coordinate the state variables will reduce 3
layer1_output_number = 200
layer2_output_number = 150
layer3_output_number = 100
layer4_output_number = 50 

# mid robots network parameter
number_of_mid_state = layer4_output_number * 2
m_layer1_output_number = 200
m_layer2_output_number = 100
m_layer3_output_number = 100
m_layer4_output_number = layer4_output_number 

def add_layer(inputs, in_size, out_size, W_name, B_name, activation_function=None):
    with tf.name_scope('layer'):
        with tf.name_scope('weights'):
            #Weights = tf.Variable(tf.random_normal([in_size, out_size]), name=W_name)
            Weights = tf.Variable(tf.zeros([in_size, out_size]), name=W_name)
        with tf.name_scope('biases'):
            #biases = tf.Variable(tf.zeros([1, out_size]) + 0.1, name=B_name)
            biases = tf.Variable(tf.zeros([1, out_size]), name=B_name)
        with tf.name_scope('Wx_plus_b'):
            Wx_plus_b = tf.matmul(inputs, Weights) + biases
        if activation_function is None:
            outputs = Wx_plus_b
        else:
            outputs = activation_function(Wx_plus_b)
        return outputs, Weights, biases

class Two_robot_network():
    def __init__(self, name):
        self.name = name
        with tf.name_scope('2_robot_State_'+self.name):
            self.state = tf.placeholder(tf.float32, [None, number_of_state])
        with tf.name_scope('2_robot_NW_'+self.name):
            self.H1, self.W1, self.B1 = add_layer(self.state, number_of_state, layer1_output_number, 'W1_12', 'B1_12', activation_function=tf.nn.relu)
            self.H2, self.W2, self.B2 = add_layer(self.H1, layer1_output_number, layer2_output_number, 'W2_12', 'B2_12', activation_function=tf.nn.relu)
            self.H3, self.W3, self.B3 = add_layer(self.H2, layer2_output_number, layer3_output_number, 'W3_12', 'B3_12', activation_function=tf.nn.relu)
            self.H4, self.W4, self.B4 = add_layer(self.H3, layer3_output_number, layer4_output_number, 'W4_12', 'B4_12', activation_function=tf.nn.sigmoid)
            self.value, self.Wf, self.Bf = add_layer(self.H4, layer4_output_number, 1, 'Wf', 'Bf', activation_function=tf.nn.tanh)
            self.two_network_saver = tf.train.Saver({'W1':self.W1,'W2':self.W2,'W3':self.W3,'W4':self.W4,'Wf':self.Wf,
                            'B1':self.B1,'B2':self.B2,'B3':self.B3,'B4':self.B4,'Bf':self.Bf})
    
    def restore_parameter(self, session, network):
        self.two_network_saver.restore(session, network)
        
class mid_network():
    def __init__(self, name, mid_half_1, mid_half_2):
        self.name = name
        with tf.name_scope('Mid_state_'+self.name):
            self.mid_state = tf.concat([mid_half_1, mid_half_2],1)

        with tf.name_scope('mid_NW_'+self.name):
            self.mH1, self.mW1, self.mB1 = add_layer(self.mid_state, number_of_mid_state, m_layer1_output_number, 'mW1', 'mB1', activation_function=tf.nn.relu)
            self.mH2, self.mW2, self.mB2 = add_layer(self.mH1, m_layer1_output_number, m_layer2_output_number, 'mW2', 'mB2', activation_function=tf.nn.relu)
            self.mH3, self.mW3, self.mB3 = add_layer(self.mH2, m_layer2_output_number, m_layer3_output_number, 'mW3', 'mB3', activation_function=tf.nn.relu)
            self.mH4, self.mW4, self.mB4 = add_layer(self.mH3, m_layer3_output_number, m_layer4_output_number, 'mW4', 'mB4', activation_function=tf.nn.sigmoid)
            self.mid_network_saver = tf.train.Saver({'mW1':self.mW1,'mW2':self.mW2,'mW3':self.mW3,'mW4':self.mW4,
                            'mB1':self.mB1,'mB2':self.mB2,'mB3':self.mB3,'mB4':self.mB4})
    
    def restore_parameter(self, session, network):
        self.mid_network_saver.restore(session, network)
        
        
class final_output():
    def __init__(self, name, final_state):
        self.name = name
        with tf.name_scope('final_state_'+self.name):
            self.final_state = final_state
        with tf.name_scope('final_output_layer'):
            self.res_value, self.mWf, self.mBf = add_layer(self.final_state, m_layer4_output_number, 1, 'mWf', 'mBf', activation_function=tf.nn.tanh)
            self.final_layer_saver = tf.train.Saver({'mWf':self.mWf,'mBf':self.mBf})
            
    def restore_parameter(self, session, network):
        self.final_layer_saver.restore(session, network)
        