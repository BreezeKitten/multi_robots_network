# -*- coding: utf-8 -*-
"""
Created on Thu Aug 27 15:17:56 2020

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
import Agent_VO as Agent
import Network
import configparser
import Combination

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
two_robot_Network_Path = '2_robot_network/gamma09_95_0429/test.ckpt'

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

Network_Path_Dict = {'2':'2_robot_network/gamma09_95_0429/test.ckpt', 
                     '3-1':'multi_robot_network/3_robot_network/0824_925/3_robot.ckpt',
                     '3-2':'multi_robot_network/3_robot_network/0826_933/3_robot.ckpt'
                     
        }




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
    
def Build_network(session, robot_num, base_network, base_network_path):
    Network_set = []
    item_list = [str(i+2) for i in range(robot_num-1)]
    Comb_list = Combination.Combination_list(item_list, base_network - 1)
    for item in Comb_list:
        name = '1'
        for i in item: name += i
        Network_set.append(Network.Network_Dict[str(base_network)](name))
    if len(Network_set) == 1:
        Target_network = Network_set[0]
        Train_Dict = {}
        Train_Dict['ref_value'] = tf.placeholder(tf.float32, [None, 1])
        Train_Dict['cost'] = tf.losses.mean_squared_error(Target_network.value, Train_Dict['ref_value'])
        Train_Dict['loss'] = Train_Dict['cost']
        Train_Dict['loss_record'] = tf.summary.scalar('loss',Train_Dict['loss'])
        Train_Dict['train_step'] = tf.train.AdamOptimizer(1e-3).minimize(Train_Dict['loss'])
        init = tf.global_variables_initializer()
        session.run(init)
        with tf.name_scope('Pred_value'):
            pred_value = Target_network.value
    
    else:
        Train_Dict = None
        with tf.name_scope('Pred_value'):
            smaller_value_list = [Network_set[0].value]
            for i in range(len(Network_set)-1):
                smaller_value_list.append(tf.minimum(smaller_value_list[i], Network_set[i+1].value))
            pred_value = smaller_value_list[-1]
            
    for item in Network_set:
        item.restore_parameter(session, base_network_path)
        
    return pred_value, Network_set, Train_Dict



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

def Set_Agent(name):
    Px = float(input('Px(-5~5m): '))
    Py = float(input('Py(-5~5m): '))
    Pth = float(input('Pth(0~6.28): '))
    V = 0 #(random.random() - 0.5) * V_max
    W = 0 #(random.random() - 0.5) * W_max
    r = float(input('r(0.1~1m): '))
    gx = float(input('gx(-5~5m): '))
    gy = float(input('gy(-5~5m): '))
    gth = float(input('gth(0~6.28): '))
    rank = int(input('rnak(1.2.3): '))
    return Agent.Agent(name, Px, Py, Pth, V, W, r, gx, gy, gth, rank, mode = 'Greedy')

def Predict_action_value(main_agent, Agent_Set, V_pred, W_pred, base_network):
    Other_Set, State_list = [], []
    VO_flag = False
    for agent in Agent_Set:
        if main_agent.name != agent.name:
            Other_Set.append(agent)
    Comb_Set = Combination.Combination_list(Other_Set, base_network-1)
    
    pred_state = main_agent.Predit_state(V_pred, W_pred, dt = deltaT)
    obs_gx, obs_gy, obs_gth = main_agent.Relative_observed_goal(pred_state.Px, pred_state.Py, pred_state.Pth)
    
    for Comb_item in Comb_Set:
        other_state = [V_pred, W_pred, main_agent.state.r, obs_gx, obs_gy, obs_gth, V_max] 
        for agent in Comb_item:
            obs_state = agent.Relative_observed_state(pred_state.Px, pred_state.Py, pred_state.Pth)            
            m11, m12, m13 = 0, 0, 0
            if main_agent.rank > agent.rank:   
                m11 = 1
            elif main_agent.rank < agent.rank:   
                m13 = 1
            else:   
                m12 = 1
            VO_flag = VO_flag or Agent.If_in_VO(pred_state, obs_state, time_factor='INF')
            other_state += [m11, m12, m13, obs_state.x, obs_state.y, obs_state.Vx, obs_state.Vy, obs_state.r]
        State_list.append([other_state])
            
    if len(State_list) == len(Network_list):
        state_dict = {}
        for i in range(len(State_list)):
            state_dict[Network_list[i].state] = State_list[i]
    else:
        print('robot num error')
        return 0
    value_matrix = sess.run(Value, feed_dict = state_dict)
    
    VO_R = 0

    if not VO_flag:
        VO_R = 0.5
    else:
        VO_R = 0
    
    
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
                
    return action_value


def Choose_action_from_Network(main_agent, Agent_Set, epsilon, base_network):
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
                action_value = Predict_action_value(main_agent, Agent_Set, V_pred, W_pred, base_network)
                if action_value > action_value_max:
                    action_value_max = action_value
                    action_pair = [V_pred, W_pred]                    
        V_pred = action_pair[0]
        W_pred = action_pair[1]
        #print(action_value_max)
    return V_pred, W_pred


def Choose_action(main_agent, Agent_Set, base_network):
    if main_agent.mode == 'Static':
        V_next, W_next = 0, 0
    if main_agent.mode == 'Random':
        V_next = main_agent.state.V + random.random() - 0.5
        W_next = main_agent.state.W + random.random() - 0.5
    if main_agent.mode == 'Greedy':
        V_next, W_next = Choose_action_from_Network(main_agent, Agent_Set, 0, base_network)
        
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

def TEST_process_all_Goal(robot_num, epsilon, RL_SAVE_PATH, base_network):    
    Main_Agent = Set_Agent('Main')
    Agent_Set = [Main_Agent]
    for i in range(robot_num-1):
        Agent_Set.append(Set_Agent(str(i+2)))      
    
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
        print('Collision or Goal error!')
        return

    if Check_Goal(Main_Agent, Calculate_distance(resX, resY, 0, 0), resTH):
        print('Initial error, the main agent at goal')
        return
    
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
                V_next, W_next = Choose_action(agent, Agent_Set, base_network)
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
    return


if __name__ == '__main__':
    NOW =  datetime.datetime.now().strftime('%Y%m%d-%H%M%S')
    print('Test!')
    print(Network_Path_Dict)    
    sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options))
    
    Robot_num = int(input('How many robots: '))
    Base_Network_path = input('Base network path: ')
    Base_Network_num = int(input('Base network num: '))
    
    print('Build Network for ', Robot_num, ' robots with ', Base_Network_num, ' robot network at ', Base_Network_path)
        
    Value, Network_list, Train = Build_network(sess, Robot_num, Base_Network_num, Base_Network_path)
    
    SAVE_PATH = input('Save Path: ')
    TEST_process_all_Goal(Robot_num, 0, SAVE_PATH, Base_Network_num)

    
