# -*- coding: utf-8 -*-
"""
Created on Tue Jul 21 15:51:25 2020

@author: BreezeCat
"""
import math as m
import matplotlib.pyplot as plt
import copy
import json

class State():
    def __init__(self, Px, Py, Pth, V, W, r):
        self.Px = Px
        self.Py = Py
        self.Pth = Pth
        self.V = V
        self.W = W
        self.r = r
    def List(self):
        return [self.Px, self.Py, self.Pth, self.V, self.W, self.r]
    
    def Record_state(self, save_file):
        file = open(save_file, 'a+')
        state = {}
        state['Px'], state['Py'], state['Pth'], state['V'], state['W'], state['r'] = self.Px, self.Py, self.Pth, self.V, self.W, self.r
        json.dump(state, file)
        file.writelines('\n') 
        file.close()
        return

class Observed_State():
    def __init__(self, x, y, Vx, Vy, r):
        self.x = x
        self.y = y
        self.Vx = Vx
        self.Vy = Vy
        self.r = r
    def List(self):
        return [self.x, self.y, self.Vx, self.Vy, self.r]

class Agent():
    def __init__(self, name, Px, Py, Pth, V, W, r, gx, gy, gth, rank, mode = 'Greedy'):
        self.name = name
        self.state = State(Px, Py, Pth, V, W, r)
        self.gx, self.gy, self.gth, self.rank, self.mode = gx, gy, gth, rank, mode
        self.Path = [copy.deepcopy(self.state)]
        self.Goal_state = 'Not'
        
    def Update_state(self, dt = 0.1):
        TH = Correct_angle(self.state.Pth +  self.state.W * dt)
        self.state.Px += self.state.V * m.cos((self.state.Pth+TH)/2) * dt
        self.state.Py += self.state.V * m.sin((self.state.Pth+TH)/2) * dt
        self.state.Pth = TH
        self.Path.append(copy.deepcopy(self.state))
        
    def Predit_state(self, V_pred, W_pred, dt = 0.1):
        TH = Correct_angle(self.state.Pth +  W_pred * dt)
        Px_pred = self.state.Px + V_pred * m.cos((self.state.Pth+TH)/2) * dt
        Py_pred = self.state.Py + V_pred * m.sin((self.state.Pth+TH)/2) * dt
        return State(Px_pred, Py_pred, TH, V_pred, W_pred, self.state.r)
    
    def Set_V_W(self, V_next, W_next):
        self.state.V = V_next
        self.state.W = W_next
        
    def Relative_observed_goal(self,  observe_x, observe_y, observe_th):
        gx_temp = self.gx - observe_x
        gy_temp = self.gy - observe_y
        gth_obs = Correct_angle(self.gth - observe_th)
        th_obs = observe_th
        gx_obs = m.cos(th_obs) * gx_temp + m.sin(th_obs) * gy_temp
        gy_obs = -m.sin(th_obs) * gx_temp + m.cos(th_obs) * gy_temp
        return gx_obs, gy_obs, gth_obs
    
    def Relative_observed_state(self, observe_x, observe_y, observe_th):
        x_temp = self.state.Px - observe_x
        y_temp = self.state.Py - observe_y
        #th_obs = Correct_angle(self.state.Pth - observe_th)
        th_obs = observe_th
        x_obs = m.cos(th_obs) * x_temp + m.sin(th_obs) * y_temp
        y_obs = -m.sin(th_obs) * x_temp + m.cos(th_obs) * y_temp
        Vx_obs = m.cos(th_obs) * self.state.V * m.cos(self.state.Pth) + m.sin(th_obs) * self.state.V * m.sin(self.state.Pth)
        Vy_obs = -m.sin(th_obs) * self.state.V * m.cos(self.state.Pth) + m.cos(th_obs) * self.state.V * m.sin(self.state.Pth)
        return Observed_State(x_obs, y_obs, Vx_obs, Vy_obs, self.state.r)
    
    def Plot_state(self, ax, color = 'b'):
        L = 0.5
        plt.plot(self.state.Px, self.state.Py, color+'o')
        plt.arrow(self.state.Px, self.state.Py, L*m.cos(self.state.Pth), L*m.sin(self.state.Pth))
        circle1 = plt.Circle( (self.state.Px, self.state.Py), self.state.r, color = color, fill = False)
        ax.add_artist(circle1)
    
    def Plot_goal(self, ax ,color = 'b'):
        L = 0.5
        plt.plot(self.gx, self.gy, color+'o')
        plt.arrow(self.gx, self.gy, L*m.cos(self.gth), L*m.sin(self.gth))
    
    def Plot_Path(self, ax, color = 'b'):
        L = 0.5       
        circle = []
        i = 0
        last_item = self.Path[0]
        for item in self.Path:
            plt.plot([last_item.Px, item.Px], [last_item.Py, item.Py], color+'-')
            if i%10 == 0:
                plt.plot(item.Px, item.Py, color+'o')
                plt.arrow(item.Px, item.Py, L*m.cos(item.Pth), L*m.sin(item.Pth))
                circle.append(plt.Circle( (item.Px, item.Py), item.r, color = color, fill = False))
                plt.text(item.Px-0.2, item.Py, str(round(i*0.1,1)), bbox=dict(color=color, alpha=0.5))
                ax.add_artist(circle[-1])
            i += 1
            last_item = item
        plt.plot(item.Px, item.Py, color+'o')
        plt.arrow(item.Px, item.Py, L*m.cos(item.Pth), L*m.sin(item.Pth))
        circle.append(plt.Circle( (item.Px, item.Py), item.r, color = color, fill = False))
        plt.text(item.Px-0.2, item.Py, str(round((i-1)*0.1,1)), bbox=dict(color=color, alpha=0.5))
        ax.add_artist(circle[-1])
        
    def Record_data(self, save_path):
        file_name = save_path + '/' + self.name + '.json'
        init_para = {}
        init_para['gx'], init_para['gy'], init_para['gth'], init_para['rank'], init_para['mode'], init_para['result'] = self.gx, self.gy, self.gth, self.rank, self.mode, self.Goal_state
        file = open(file_name, 'a+')
        json.dump(init_para, file)
        file.writelines('\n') 
        file.close()
        for state in self.Path:
            state.Record_state(file_name)
        
    
def Correct_angle(angle):
    angle = m.fmod(angle, 2*m.pi)
    if angle < 0:
        angle = angle + 2*m.pi
    return angle



def main_test():
    A = Agent('A',1,1,0,1,1,0.5,0,0,0,1)
    B = Agent('B',-1,-1,0,1,-1,0.2,0,0,0,2)
    ax = plt.gca()
    ax.set_xlim((-5,5))
    ax.set_ylim((-5,5))
    A.Plot_state(ax = ax)
    B.Plot_state(ax = ax, color = 'r')
    plt.savefig('test.png')
    plt.close('all')
    ax = plt.gca()
    ax.set_xlim((-5,5))
    ax.set_ylim((-5,5))
    for i in range(45):
        A.Update_state(0.1)
        B.Update_state(0.1)
    A.Plot_Path(ax = ax)
    B.Plot_Path(ax = ax, color='r')
    A.Record_data('logs')