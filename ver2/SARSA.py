# -*- coding: utf-8 -*-
"""
Created on Tue Dec 27 13:33:05 2022

@author: parkh
"""
    
import pandas as pd
import numpy as np
from datetime import datetime
import plotly.express as px
import copy
import random
from matplotlib import pylab as plt
from Resource import *
from Job import *
from collections import defaultdict
from FJSP_SIMULATOR2 import *

class QAgent():
    def __init__(self):
        self.q_table = np.zeros((100000000,6))
        self.eps = 0.9
        self.alpha = 0.1
    
    def get_state(self, s):
        state = int(s)
        return state
        
    def select_action(self, s):
        coin = random.random()
        k = self.get_state(s)
        if coin < self.eps:
            action = random.randint(0,5)
        else:
            action_val = copy.deepcopy(self.q_table[k,:])
            action = np.argmax(action_val)
        return action
    
    def select_action2(self, s):
        k = self.get_state(s)
        action_val = self.q_table[k,:]
        action = np.argmax(action_val)
        return action
    
    def update_table(self, transition):
        s, a, r, s_prime = transition
        k = self.get_state(s)
        next_k = s_prime
        a_prime = self.select_action(s_prime) 
        next_k=self.get_state(next_k)
        self.q_table[k,a] = self.q_table[k,a] + self.alpha * (r + self.q_table[next_k, a_prime] - self.q_table[k,a])
            
    def anneal_eps(self):
        self.eps -=0.001
        self.eps = max(self.eps, 0.2)
    
    def show(self):
        print(self.q_table.tolist())
        print(self.eps)
      #리워드 수정
      #num_of_op
      #플로우타임 고려
      #깃허브에 정리
      #q_table -> 신경망 전환
def main():
    env = FJSP_simulator('C:/Users/parkh/FJSP_SIM4.csv','C:/Users/parkh/FJSP_SETUP_SIM.csv',1)
    agent = QAgent()
    for n_epi in range(1000):
        print(n_epi)
        done = False
        s = env.reset()
        z=0
        while not done:
            a = agent.select_action(s)
            s_prime, r, done2 = env.step(a)
            if done2 == False:
                z+=1
            if done2 == False:
                agent.update_table((s,a,r,s_prime))
                s = s_prime
            else:
                done = env.process_event()
        agent.anneal_eps()
        #if n_epi%10==0 or n_epi<10:
            #print(n_epi,"에피소드가 지남")
            #agent.show()
    
    done=False
    s=env.reset()
    k=[]
    while not done:
        a = agent.select_action2(s)
        k.append(a)
        s_prime, r,done2 = env.step(a)
        if done2 == True:
            done = env.process_event()
        else:
            print(s)
            print(a)
            s = s_prime
    Flow_time, machine_util, util, makespan = env.performance_measure()
    print(len(k))
    #agent.show()
    return Flow_time, machine_util, util, makespan
av=0
for i in range(1):
    Flow_time, machine_util, util, makespan=main()
    print("FlowTime:" , Flow_time)
    print("machine_util:" , machine_util)
    print("util:" , util)
    print("makespan:" , makespan)