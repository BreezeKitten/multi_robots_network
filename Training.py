# -*- coding: utf-8 -*-
"""
Created on Thu Aug 13 15:23:08 2020

@author: BreezeCat
"""

import sys
import tensorflow as tf
import json
import Network
import random
import numpy as np

tf.reset_default_graph()
gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.20)

'''
DL Parameter
'''
training_eposide_num = 5000 #100000 
training_num = 3000 #1500 #3000
test_num = 1

robot_state_list_3 = ['V', 'W', 'r1', 'gx', 'gy', 'gth', 'Vmax', 'm11_2', 'm12_2', 'm13_2', 'Px2', 'Py2', 'Vx2', 'Vy2', 'r2', 'm11_3', 'm12_3', 'm13_3', 'Px3', 'Py3', 'Vx3', 'Vy3', 'r3']
robot_state_list_4 = ['V', 'W', 'r1', 'gx', 'gy', 'gth', 'Vmax', 'm11_2', 'm12_2', 'm13_2', 'Px2', 'Py2', 'Vx2', 'Vy2', 'r2', 'm11_3', 'm12_3', 'm13_3', 'Px3', 'Py3', 'Vx3', 'Vy3', 'r3', 'm11_4', 'm12_4', 'm13_4', 'Px4', 'Py4', 'Vx4', 'Vy4', 'r4']

def Read_data(file_name):
    data = {}
    file = open(file_name,'r')
    data_line = file.readline()
    count = 0
    while(data_line):
        data[count] = json.loads(data_line)
        data_line = file.readline()
        count = count + 1
    file.close()
    return data

def Sample_data(data_base, sample_number):
    sampled_data = {}
    sample_array = random.sample(range(0,len(data_base)), sample_number)
    for index in sample_array:
        sampled_data[index] = data_base[index]
    return sampled_data

def Divide_state_value(data, state_list):
    Start_flag = 1
    for item in data:
        small_temp_state = []
        for key in state_list:
            small_temp_state.append(data[item][key])
        temp_state = [small_temp_state]
        temp_value = [[data[item]['Value']]]
        if Start_flag:
            state = temp_state
            value = temp_value
            Start_flag = 0
        else:
            state = np.concatenate((state, temp_state), axis=0)
            value = np.concatenate((value, temp_value), axis=0)
    return state, value

def DL_process(sess, Target_network, DL_database, state_list, train_step, loss_record, Network, value, writer):
    data = Read_data(DL_database)
    Target_network.network_saver.save(sess, Network)   
    for training_eposide in range(training_eposide_num):
        training_data = Sample_data(data, training_num)
        training_state, training_value = Divide_state_value(training_data, state_list)
        sess.run(train_step, feed_dict={Target_network.state: training_state, value: training_value})        

        if training_eposide%100 == 0:
            rs = sess.run(loss_record, feed_dict = {Target_network.state: training_state, value: training_value})
            writer.add_summary(rs, training_eposide)
            print('record', training_eposide)
            Target_network.network_saver.save(sess, Network)   

    Target_network.network_saver.save(sess, Network)    
    return


if __name__ == '__main__':
    Network_path = 'TEST/Network/'
    DL_log_path = 'TEST/DL_log'
    Network_file_name = '3_robot.ckpt'
    
    sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options))
    Target_network = Network.Three_robot_network('train')
    
    value = tf.placeholder(tf.float32, [None, 1])
    cost = tf.losses.mean_squared_error(Target_network.value, value)
    regularizers = tf.nn.l2_loss(Target_network.W1) + tf.nn.l2_loss(Target_network.W2) + tf.nn.l2_loss(Target_network.W3) + tf.nn.l2_loss(Target_network.W4) + tf.nn.l2_loss(Target_network.Wf)
    loss = cost + 0.0001* regularizers
    
    loss_record = tf.summary.scalar('loss',loss)
    
    train_step = tf.train.AdamOptimizer(1e-3).minimize(loss)
  
    
    init = tf.global_variables_initializer()
    sess.run(init)       
    #Target_network.restore_parameter(sess, Network_path)
    writer = tf.summary.FileWriter(DL_log_path, sess.graph)
