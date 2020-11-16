# -*- coding: utf-8 -*-
"""
Created on Wed Aug  5 12:34:20 2020

@author: BreezeCat
"""


import matplotlib.pyplot as plt
import Agent
import json
import copy
import math
import numpy as np
import tkinter as tk
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import time
import threading

color_list = ['b', 'g', 'r', 'c', 'm', 'y', 'k']
x_upper_bound = 5       #unit:m
x_lower_bound = -5      #unit:m
y_upper_bound = 5       #unit:m
y_lower_bound = -5      #unit:m

Arrived_reward = 1
Time_out_penalty = -0.25
Collision_high_penalty = -0.5
Collision_low_penalty = -1
Collision_equ_penalty = -0.75
Reward_dict = {}
Reward_dict['Finish'], Reward_dict['TIME_OUT'] = Arrived_reward, Time_out_penalty  
Reward_dict['Collision_low'], Reward_dict['Collision_high'], Reward_dict['Collision_equal'] = Collision_high_penalty, Collision_low_penalty, Collision_equ_penalty
gamma = 0.9

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
    
    
def Calculate_Value(data, result):
    for count in data:
        for item in data[count]:
            if int(count) <= result[item][0]:
                data[count][item]['Value'] = Reward_dict[result[item][1]] * math.pow(gamma, result[item][0]-int(count))
    return data

def Save_data(data, result, save_path):
    file = open(save_path, 'a+')
    for count in data:
        for item in data[count]:
            if int(count) <= result[item][0]:
                json.dump(data[count][item], file)
                file.writelines('\n')
    file.close()           
    return data


def Transform_learning_data(log_path, robot_num, save_path):
    robot_dict = Open_logs(log_path, robot_num)
    count = 0
    data = {}
    result = {}
    while(1):
        data[str(count)] = {}
        for main in robot_dict:
            main_agent = robot_dict[main]['Agent']
            main_state = main_agent.state
            obs_gx, obs_gy, obs_gth = main_agent.Relative_observed_goal(main_state.Px, main_state.Py, main_state.Pth)
            data[str(count)][main_agent.name] = {}
            dataline = data[str(count)][main_agent.name]
            dataline['V'], dataline['W'], dataline['r1'], dataline['Vmax'], dataline['gx'], dataline['gy'], dataline['gth'] = main_state.V, main_state.W, main_state.r, 3, obs_gx, obs_gy, obs_gth
            obs_robot_num = 1
            for obs in robot_dict:
                obs_agent = robot_dict[obs]['Agent']
                if main_agent.name != obs_agent.name:
                    obs_robot_num += 1
                    obs_state = obs_agent.Relative_observed_state(main_state.Px, main_state.Py, main_state.Pth)                    
                    m11, m12, m13 = 0, 0, 0
                    if main_agent.rank > obs_agent.rank:   
                        m11 = 1
                    elif main_agent.rank < obs_agent.rank:   
                        m13 = 1
                    else:   
                        m12 = 1
                    num = str(obs_robot_num)
                    dataline['Px'+num], dataline['Py'+num], dataline['Vx'+num], dataline['Vy'+num], dataline['r'+num] = obs_state.x, obs_state.y, obs_state.Vx, obs_state.Vy, obs_state.r
                    dataline['m11_'+num], dataline['m12_'+num], dataline['m13_'+num] =  m11, m12, m13
            #print(dataline)
                
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
            for item in robot_dict:
                agent = robot_dict[item]['Agent']
                if agent.name not in result:
                    result[agent.name] = [count+1, robot_dict[item]['result']]
            break
        else:
            for item in robot_dict:
                agent = robot_dict[item]['Agent']
                if agent.Path[-2].Px == agent.Path[-1].Px and agent.Path[-2].Py == agent.Path[-1].Py and agent.Path[-2].Pth == agent.Path[-1].Pth:
                    if agent.Path[-2].V != agent.Path[-1].V and agent.Path[-2].W != agent.Path[-1].W:
                        result[agent.name] = [count+1, robot_dict[item]['result']]
            
            count += 1
    Close_logs(robot_dict)
    
    data = Calculate_Value(data, result)
    Save_data(data, result, save_path)
    return data, result
    
def Read_log_list(log_list_file):
    file = open(log_list_file, 'r', encoding="utf-8")
    data = file.readline()
    log_list = []
    while(data):
        log_list.append(data[:-1])
        data = file.readline()
    file.close()
    return log_list


def GIF_process_TK(log_path, robot_num, TKapp):
    robot_dict = Open_logs(log_path, robot_num)
    count = 0
    while(1):
        #plt.close('all')
        fig = plt.figure(figsize=(9,9))
        #ax = plt.gca()
        ax = fig.add_subplot(1, 1, 1)
        
        
        
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
        #plt.savefig(save_path +'/'+ str(count).zfill(4) +'.png')
        fig_tk = FigureCanvasTkAgg(fig, TKapp)
        fig_tk.get_tk_widget().grid(column=0, row=10, ipadx=5, pady=5, sticky=tk.W+tk.N)
        time.sleep(0.1)
        print(count)
        
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


def callback(log_path, robot_num, TKapp):
    #GIF_process_TK(log_path, robot_num, TKapp)
    a = threading.Thread(target=GIF_process_TK, args=(log_path, robot_num, TKapp))
    a.daemon = True
    a.start()
    return


            
def Window_app(log_list_file, robot_num):
    log_list = Read_log_list(log_list_file)  
    app = tk.Tk()
    app.geometry('920x920')
    variable = tk.StringVar(app)
    variable.set(log_list[0])
    opt = tk.OptionMenu(app, variable, *log_list)
    opt.config(width=90, font=('Helvetica', 12))
    #opt.pack()
    opt.grid()
    
    variable.trace("w", lambda *args: callback(variable.get(), robot_num, app))
    app.mainloop()




if __name__ == '__main__':
    print('Replay log')
    
        
