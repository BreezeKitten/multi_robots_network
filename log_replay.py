# -*- coding: utf-8 -*-
"""
Created on Wed Aug  5 12:34:20 2020

@author: BreezeCat
"""
import matplotlib.pyplot as plt
import Agent
import json
import copy

color_list = ['b', 'g', 'r', 'c', 'm', 'y', 'k']
x_upper_bound = 5       #unit:m
x_lower_bound = -5      #unit:m
y_upper_bound = 5       #unit:m
y_lower_bound = -5      #unit:m


def Open_logs(log_path, robot_num):
    print('Open logs in ' + log_path + '\nfor "' + str(robot_num) + '" robots')
    name_list = ['Main'] + [str(i) for i in range(2,robot_num+1)]
    log_dict = {}
    for item in name_list:
        log_dict[item] = {}
        log_dict[item]['log'] = open(log_path + '/' + item + '.json')
        goal_state = json.loads(log_dict[item]['log'].readline())
        init_state = json.loads(log_dict[item]['log'].readline())
        log_dict[item]['Agent'] = Agent.Agent(item, init_state['Px'], init_state['Py'], init_state['Pth'], init_state['V'], init_state['W'], init_state['r'], 
                goal_state['gx'], goal_state['gy'], goal_state['gth'], goal_state['rank'], goal_state['mode'])
        log_dict[item]['result'] = goal_state['result']
    return log_dict
    

def Close_logs(log_dict):
    for item in log_dict:
        log_dict[item]['log'].close()
        
def Update_state_from_log(agent, update_state):
    agent.state.Px, agent.state.Py, agent.state.Pth, agent.state.V, agent.state.W = update_state['Px'], update_state['Py'], update_state['Pth'], update_state['V'], update_state['W']
    agent.Path.append(copy.deepcopy(agent.state))
    

def GIF_process(log_path, robot_num, save_path):
    robot_dict = Open_logs(log_path, robot_num)
    count = 0
    while(1):
        plt.close('all')
        plt.figure(figsize=(12,12))
        ax = plt.gca()
        ax.cla()    
        ax.set_xlim((x_lower_bound,x_upper_bound))     #上下限
        ax.set_ylim((x_lower_bound,x_upper_bound))
        plt.xlabel('X(m)')
        plt.ylabel('Y(m)')
        color_count = 0
        for item in robot_dict:
            robot_dict[item]['Agent'].Plot_Path(ax = ax, color = color_list[color_count%len(color_list)])
            robot_dict[item]['Agent'].Plot_goal(ax = ax, color = color_list[color_count%len(color_list)])
            color_count += 1           
        plt.savefig(save_path +'/'+ str(count).zfill(4) +'.png')
        Error = False
        for item in robot_dict:
            try:
                Update_state_from_log(robot_dict[item]['Agent'], json.loads(robot_dict[item]['log'].readline()))
            except Exception as e:   
                print('error - msg:',e) 
                Error = True
                break
        if Error:
            print('End')
            break
        else:
            count += 1
    Close_logs(robot_dict)

if __name__ == '__main__':
    print('Replay log')
    
        
