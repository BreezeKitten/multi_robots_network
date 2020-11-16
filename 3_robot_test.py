# -*- coding: utf-8 -*-
"""
Created on Thu Aug 13 16:26:08 2020

@author: BreezeCat
"""

import sys
import tensorflow as tf
import json
import random
import numpy as np
import math
import matplotlib.pyplot as plt
import datetime
import copy
import file_manger
import state_load as SL
import os
import Agent
import Network
import configparser
import log_replay 
import Training

tf.reset_default_graph()
gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.20) 
color_list = ['b', 'g', 'r', 'c', 'm', 'y', 'k']

####
#Common parameter
####
PI = math.pi
resX = 0.1 # resolution of X
resY = 0.1 # resolution of Y
resTH = PI/15
LOG_DIR = 'logs/Multi_test'
SOME_TAG = '_test4'

####
#Reward
####
Arrived_reward = 1
Time_out_penalty = -0.25
Collision_high_penalty = -0.5
Collision_low_penalty = -1
Collision_equ_penalty = -0.75


'''
DL Parameter
'''
training_eposide_num = 5000 #100000 
training_num = 1500 #3000
test_num = 1
three_robot_Network_Path = 'TEST/Network/3_robot.ckpt'

'''
Motion Parameter
'''
deltaT = 0.1            #unit:s
V_max = 3               #m/s
W_max = 2               #rad/s
linear_acc_max = 10     #m/s^2
angular_acc_max = 7     #rad/s^2
size_min = 0.1          #unit:m
x_upper_bound = 5       #unit:m
x_lower_bound = -5      #unit:m
y_upper_bound = 5       #unit:m
y_lower_bound = -5      #unit:m
TIME_OUT_FACTOR = 4


RL_eposide_num = 100
RL_epsilon = 0
gamma = 0.9




def Load_Config(file):
    print('Load config from ' + file)
    config = configparser.ConfigParser()
    config.read(file)
    configDict = {section: dict(config.items(section)) for section in config.sections()}
    print(configDict)
    return configDict

def Set_parameter(paraDict):
    global deltaT, V_max, W_max, linear_acc_max, angular_acc_max, size_min, TIME_OUT_FACTOR
    print('Set parameter\n', paraDict)
    deltaT = float(paraDict['deltat'])
    V_max, W_max, linear_acc_max, angular_acc_max = float(paraDict['v_max']), float(paraDict['w_max']), float(paraDict['linear_acc_max']), float(paraDict['angular_acc_max'])
    size_min = float(paraDict['size_min'])
    TIME_OUT_FACTOR = float(paraDict['time_out_factor'])
    
def Build_network(session, robot_num):
    '''
    Network_set = []
    for i in range(robot_num - 1):
        Network_set.append(Network.three_robot_network(str(10+i+1)))
    for item in Network_set:
        item.restore_parameter(session, three_robot_Network_Path)
    with tf.name_scope('Smallest_value'):
        smaller_value_list = [Network_set[0].value]
        for i in range(len(Network_set)-1):
            smaller_value_list.append(tf.minimum(smaller_value_list[i], Network_set[i+1].value))
        smallest_value = smaller_value_list[-1]
    '''
    Network_set = [Network.Three_robot_network('123')]
    
    for item in Network_set:
        item.restore_parameter(session, three_robot_Network_Path)
    smallest_value = Network_set[0].value
    return smallest_value, Network_set



def Calculate_distance(x1, y1, x2, y2):
    return np.sqrt(math.pow( (x1-x2) , 2) + math.pow( (y1-y2) , 2))

def Check_Collision(agent1, agent2):
    distance = Calculate_distance(agent1.state.Px, agent1.state.Py, agent2.state.Px, agent2.state.Py)
    if (distance <= (agent1.state.r + agent2.state.r)):
        return True
    else:
        return False


def Check_Goal(agent, position_tolerance, orientation_tolerance):    
    position_error = Calculate_distance(agent.state.Px, agent.state.Py, agent.gx, agent.gy)
    orientation_error = abs(agent.state.Pth - agent.gth)
    if (position_error < position_tolerance) and (orientation_error < orientation_tolerance):
        return True
    else:
        return False

def Random_Agent(name):
    Px = random.random()*(x_upper_bound - x_lower_bound) + x_lower_bound
    Py = random.random()*(y_upper_bound - y_lower_bound) + y_lower_bound
    Pth = random.random()*2*PI 
    V = 0 #(random.random() - 0.5) * V_max
    W = 0 #(random.random() - 0.5) * W_max
    r = random.random() + size_min
    gx = random.random()*(x_upper_bound - x_lower_bound) + x_lower_bound
    gy = random.random()*(y_upper_bound - y_lower_bound) + y_lower_bound
    gth = random.random()*2*PI 
    rank = random.randint(1,3)
    return Agent.Agent(name, Px, Py, Pth, V, W, r, gx, gy, gth, rank, mode = 'Greedy')

def Predict_action_value(main_agent, Agent_Set, V_pred, W_pred):
    State_list = []
    for agent in Agent_Set:
        if main_agent.name != agent.name:
            pred_state = main_agent.Predit_state(V_pred, W_pred, dt = deltaT)
            obs_state = agent.Relative_observed_state(pred_state.Px, pred_state.Py, pred_state.Pth)
            obs_gx, obs_gy, obs_gth = main_agent.Relative_observed_goal(pred_state.Px, pred_state.Py, pred_state.Pth)
            m11, m12, m13 = 0, 0, 0
            if main_agent.rank > agent.rank:   
                m11 = 1
            elif main_agent.rank < agent.rank:   
                m13 = 1
            else:   
                m12 = 1
            for agent2 in Agent_Set:
                if main_agent.name != agent2.name and agent.name != agent2.name:
                    obs_state_2 = agent2.Relative_observed_state(pred_state.Px, pred_state.Py, pred_state.Pth)
                    m11_2, m12_2, m13_2 = 0, 0, 0
                    if main_agent.rank > agent2.rank:   
                        m11_2 = 1
                    elif main_agent.rank < agent2.rank:   
                        m13_2 = 1
                    else:   
                        m12_2 = 1
                    State_list.append([[V_pred, W_pred, main_agent.state.r, obs_gx, obs_gy, obs_gth, V_max, m11, m12, m13, obs_state.x, obs_state.y, obs_state.Vx, obs_state.Vy, obs_state.r, m11_2, m12_2, m13_2, obs_state_2.x, obs_state_2.y, obs_state_2.Vx, obs_state_2.Vy, obs_state_2.r]])
    
    if len(State_list) == len(Network_list)*2:
        state_dict = {}
        for i in range(len(Network_list)):
            state_dict[Network_list[i].state] = State_list[i]
            #print(State_list[i])
    else:
        print('robot num error')
        return 0
    value_matrix = sess.run(Value, feed_dict = state_dict)
    
    R = 0
    
    main_agent_pred = Agent.Agent('Pred', pred_state.Px, pred_state.Py, pred_state.Pth, pred_state.V, pred_state.W, pred_state.r, main_agent.gx, main_agent.gy, main_agent.gth, main_agent.rank)
    if Check_Goal(main_agent_pred, Calculate_distance(resX, resY, 0, 0), resTH):
        R = Arrived_reward
    for item in Agent_Set:
        if main_agent.name != item.name:
            if Check_Collision(main_agent, item):
                if main_agent.rank > item.rank:
                    R = Collision_high_penalty
                elif main_agent.rank < item.rank:   
                    R = Collision_low_penalty
                else:   
                    R = Collision_equ_penalty
                break
    action_value = R + value_matrix[0][0]
    #print(action_value)            
    return action_value


def Choose_action_from_Network(main_agent, Agent_Set, epsilon):
    dice = random.random()
    action_value_max = -999999   
    if dice < epsilon:
        linear_acc = -linear_acc_max + random.random() * 2 * linear_acc_max
        angular_acc = -angular_acc_max + random.random() * 2 * angular_acc_max
        V_pred = np.clip(main_agent.state.V + linear_acc * deltaT, -V_max, V_max)
        W_pred = np.clip(main_agent.state.W + angular_acc * deltaT, -W_max, W_max)
    else:
        linear_acc_set = np.arange(-linear_acc_max, linear_acc_max, 1)
        angular_acc_set = np.arange(-angular_acc_max, angular_acc_max, 1)
        for linear_acc in linear_acc_set:
            V_pred = np.clip(main_agent.state.V + linear_acc * deltaT, -V_max, V_max)
            for angular_acc in angular_acc_set:
                W_pred = np.clip(main_agent.state.W + angular_acc * deltaT, -W_max, W_max)
                action_value = Predict_action_value(main_agent, Agent_Set, V_pred, W_pred)
                if action_value > action_value_max:
                    action_value_max = action_value
                    action_pair = [V_pred, W_pred]                    
        V_pred = action_pair[0]
        W_pred = action_pair[1]
        #print(action_value_max)
    return V_pred, W_pred


def Choose_action(main_agent, Agent_Set):
    if main_agent.mode == 'Static':
        V_next, W_next = 0, 0
    if main_agent.mode == 'Random':
        V_next = main_agent.state.V + random.random() - 0.5
        W_next = main_agent.state.W + random.random() - 0.5
    if main_agent.mode == 'Greedy':
        V_next, W_next = Choose_action_from_Network(main_agent, Agent_Set, 0)
        
    return V_next, W_next

def Show_Path(Agent_Set, result, save_path):
    plt.close('all')
    plt.figure(figsize=(12,12))
    ax = plt.gca()
    ax.cla()    
    ax.set_xlim((x_lower_bound,x_upper_bound))     #上下限
    ax.set_ylim((x_lower_bound,x_upper_bound))
    plt.xlabel('X(m)')
    plt.ylabel('Y(m)')
    color_count = 0
    for agent in Agent_Set:
        agent.Plot_Path(ax = ax, color = color_list[color_count%len(color_list)])
        agent.Plot_goal(ax = ax, color = color_list[color_count%len(color_list)])
        color_count += 1
    NOW = datetime.datetime.now().strftime('%Y%m%d-%H%M%S')
    plt.savefig(save_path +'/'+ NOW + result +'.png')
    return

def RL_process(robot_num, eposide_num, epsilon, RL_SAVE_PATH):      
    for eposide in range(eposide_num):
        
        if eposide%20 == 0:
            print(eposide)
        Main_Agent = Random_Agent('Main')
        Agent_Set = [Main_Agent]
        for i in range(robot_num-1):
            Agent_Set.append(Random_Agent(str(i+2)))      
        
        time = 0
        result = 'Finish'
        
        Collision_Flag = False
        Goal_dist_Flag = False
        for item in Agent_Set:
            for item2 in Agent_Set:
                if item.name != item2.name:
                    Collision_Flag = Collision_Flag or Check_Collision(item, item2)
                    Goal_dist_Flag = Goal_dist_Flag or Calculate_distance(item.gx, item.gy, item2.gx, item2.gy) < (item.state.r + item2.state.r)
                if Collision_Flag or Goal_dist_Flag:
                    break
            if Collision_Flag or Goal_dist_Flag:
                break
        if Collision_Flag or Goal_dist_Flag:
            continue

        if Check_Goal(Main_Agent, Calculate_distance(resX, resY, 0, 0), resTH):
            continue
          
        TIME_OUT = Calculate_distance(Main_Agent.state.Px, Main_Agent.state.Py, Main_Agent.gx, Main_Agent.gy) * TIME_OUT_FACTOR
        
        
        NOW = datetime.datetime.now().strftime('%Y%m%d-%H%M%S')
        save_path = RL_SAVE_PATH + '/' + NOW
        os.makedirs(save_path)
        while(not Check_Goal(Main_Agent, Calculate_distance(resX, resY, 0, 0), resTH)):
            if time > TIME_OUT:
                result = 'TIME_OUT'
                break
            
            for item in Agent_Set:
                if Main_Agent.name != item.name:
                    if Check_Collision(Main_Agent, item):
                        if Main_Agent.rank > item.rank:
                            result = 'Collision_high'
                        elif Main_Agent.rank < item.rank:   
                            result = 'Collision_low'
                        else:   
                            result = 'Collision_equal'
                        break
            if result != 'Finish':
                break
            else:
                for agent in Agent_Set:
                    if Check_Goal(agent, Calculate_distance(resX, resY, 0, 0), resTH):
                        V_next, W_next = 0, 0                    
                    else:
                        V_next, W_next = Choose_action(agent, Agent_Set)
                    agent.Set_V_W(V_next, W_next)
                    
                for agent in Agent_Set:
                    agent.Update_state(dt = deltaT)
                                
            time = time + deltaT
        
        
        for agent in Agent_Set:
            agent.Record_data(save_path)
        Show_Path(Agent_Set, result, RL_SAVE_PATH)
        
    return


def RL_process_all_Goal(robot_num, eposide_num, epsilon, RL_SAVE_PATH):    
    for eposide in range(eposide_num):
        if eposide%20 == 0:
            print(eposide)
        Main_Agent = Random_Agent('Main')
        Agent_Set = [Main_Agent]
        for i in range(robot_num-1):
            Agent_Set.append(Random_Agent(str(i+2)))      
        
        time = 0
        
        Collision_Flag = False
        Goal_dist_Flag = False
        for item in Agent_Set:
            for item2 in Agent_Set:
                if item.name != item2.name:
                    Collision_Flag = Collision_Flag or Check_Collision(item, item2)
                    Goal_dist_Flag = Goal_dist_Flag or Calculate_distance(item.gx, item.gy, item2.gx, item2.gy) < (item.state.r + item2.state.r)
                if Collision_Flag or Goal_dist_Flag:
                    break
            if Collision_Flag or Goal_dist_Flag:
                break
        if Collision_Flag or Goal_dist_Flag:
            continue

        if Check_Goal(Main_Agent, Calculate_distance(resX, resY, 0, 0), resTH):
            continue
        
        TIME_OUT = 0
        for agent in Agent_Set:
            TIME_OUT = max(TIME_OUT, Calculate_distance(agent.state.Px, agent.state.Py, agent.gx, agent.gy) * TIME_OUT_FACTOR)
   
       
        terminal_flag = True
        for agent in Agent_Set:
            small_goal_flag = Check_Goal(agent, Calculate_distance(resX, resY, 0, 0), resTH)
            if small_goal_flag:
                agent.Goal_state = 'Finish'
            terminal_flag = terminal_flag and small_goal_flag
            
        NOW = datetime.datetime.now().strftime('%Y%m%d-%H%M%S')
        save_path = RL_SAVE_PATH + '/' + NOW
        os.makedirs(save_path)
        while(not terminal_flag):       
            for agent1 in Agent_Set:               
                for agent2 in Agent_Set:
                    if agent1.name != agent2.name:
                        if Check_Collision(agent1, agent2):
                            if agent1.rank > agent2.rank:
                                if agent1.Goal_state == 'Not':
                                    agent1.Goal_state = 'Collision_high'
                                if agent2.Goal_state == 'Not':
                                    agent2.Goal_state = 'Collision_low'
                            elif agent1.rank < agent2.rank:
                                if agent1.Goal_state == 'Not':
                                    agent1.Goal_state = 'Collision_low'
                                if agent2.Goal_state == 'Not':
                                    agent2.Goal_state = 'Collision_high'
                            else:
                                if agent1.Goal_state == 'Not':
                                    agent1.Goal_state = 'Collision_equal'
                                if agent2.Goal_state == 'Not':
                                    agent2.Goal_state = 'Collision_equal'
                if Check_Goal(agent1, Calculate_distance(resX, resY, 0, 0), resTH) and agent1.Goal_state == 'Not':
                    agent1.Goal_state = 'Finish'


            terminal_flag = True
            for agent in Agent_Set:
                if agent.Goal_state == 'Not':
                    V_next, W_next = Choose_action(agent, Agent_Set)
                else:
                    V_next, W_next = 0, 0  
                agent.Set_V_W(V_next, W_next)
                terminal_flag = terminal_flag and agent.Goal_state != 'Not'
                       
            if time > TIME_OUT:
                for agent in Agent_Set:
                    if agent.Goal_state == 'Not':
                        agent.Goal_state = 'TIME_OUT'
                break
            
            for agent in Agent_Set:
                agent.Update_state(dt = deltaT)                        
            time = time + deltaT
        
        result = ''
        for agent in Agent_Set:
            result = result + agent.Goal_state[0]
            agent.Record_data(save_path)
        Show_Path(Agent_Set, result, save_path)
        
        log_replay.Transform_learning_data(save_path, robot_num, DL_Database)
    return



if __name__ == '__main__':
    NOW =  datetime.datetime.now().strftime('%Y%m%d-%H%M%S')
    DL_Database = 'TEST/DataBase3.json'
    #robot_state_list_3 = ['V', 'W', 'r1', 'gx', 'gy', 'gth', 'Vmax', 'm11_2', 'm12_2', 'm13_2', 'Px2', 'Py2', 'Vx2', 'Vy2', 'r2', 'm11_3', 'm12_3', 'm13_3', 'Px3', 'Py3', 'Vx3', 'Vy3', 'r3']
    '''
    if len(sys.argv) < 2:
        Configfile = input('Config file at:')
    else:
        Configfile = sys.argv[1]
    Config_dict = Load_Config(Configfile)

    sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options))
    Value, Network_list = Build_network(sess, int(Config_dict['main']['robot_num']))

    if int(Config_dict['main']['custom_parameter']):
        Set_parameter(Config_dict['parameter'])
    
    if int(Config_dict['main']['all_goal']):
        print('All goal process')
        save_path = Config_dict['main']['save_path'] + '/' + NOW +'_all_goal'
        os.makedirs(save_path)
        RL_process_all_Goal(int(Config_dict['main']['robot_num']), int(Config_dict['main']['eposide_num']), epsilon = 1, RL_SAVE_PATH = save_path)
    else:
        save_path = Config_dict['main']['save_path'] + '/' + NOW +'_main_goal'
        os.makedirs(save_path)
        RL_process(int(Config_dict['main']['robot_num']), int(Config_dict['main']['eposide_num']), epsilon = 1, RL_SAVE_PATH = save_path)
    '''
    sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options))
    Value, Network_list = Build_network(sess, 3)
    
    Target_network = Network_list[0]
    
    
    value = tf.placeholder(tf.float32, [None, 1])
    cost = tf.losses.mean_squared_error(Target_network.value, value)
    regularizers = tf.nn.l2_loss(Target_network.W1) + tf.nn.l2_loss(Target_network.W2) + tf.nn.l2_loss(Target_network.W3) + tf.nn.l2_loss(Target_network.W4) + tf.nn.l2_loss(Target_network.Wf)
    loss = cost + 0.0001* regularizers
    
    loss_record = tf.summary.scalar('loss',loss)
    
    train_step = tf.train.AdamOptimizer(1e-3).minimize(loss)
    init = tf.global_variables_initializer()
    sess.run(init) 
    Target_network.restore_parameter(sess, three_robot_Network_Path)
    writer = tf.summary.FileWriter('TEST/DL_log', sess.graph)
    #RL_process_all_Goal(3, 100, 0, 'TEST/Log/1')
    
    Test_Time = 240
    while(Test_Time<280):
        os.makedirs('TEST/Log/'+str(Test_Time))
        Training.DL_process(sess, Target_network, DL_Database, Training.robot_state_list_3, train_step, loss_record, three_robot_Network_Path, value, writer)
        if Test_Time < 260:
            e = 0.1
        else:
            e = 0
        RL_process_all_Goal(3, 100, e, 'TEST/Log/'+str(Test_Time))
        Test_Time += 1