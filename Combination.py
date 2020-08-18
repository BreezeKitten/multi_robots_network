# -*- coding: utf-8 -*-
"""
Created on Mon Aug 17 16:26:07 2020

@author: BreezeCat
"""



def Combination_list(item_list, comb_num):
    binary_list, comb_binary_list, comb_item_list = [], [], []
    for i in range(pow(2,len(item_list))):
        binary_list.append([int(j) for j in ('{0:0'+str(len(item_list))+'b}').format(i)])
        binary_list[-1].append(sum(binary_list[-1]))
    for item in binary_list:
        if item[-1] == comb_num:
            comb_binary_list.append(item[:-1])
    for item in comb_binary_list:
        comb_item = []
        for i in range(len(item)):
            if item[i]:
                comb_item.append(item_list[i])
        if comb_item != []:
            comb_item_list.append(comb_item)             
             
    return comb_item_list