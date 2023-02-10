# -*- coding: utf-8 -*-
"""
Created on Mon Apr  4 23:28:32 2022

@author: parkh
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Categorical
import gym
import collections
import random
from SIM5 import *

learning_rate = 0.0005  
gamma = 1
buffer_limit = 10000
batch_size = 16

class ReplayBuffer():
    def __init__(self):
        self.buffer = collections.deque(maxlen=buffer_limit);
    def put(self, transition):
        self.buffer.append(transition)
    def sample(self, n):
        mini_batch = random.sample(self.buffer, n)
        s_lst, a_lst, r_lst, s_prime_lst, done_mask_lst = [],[],[],[],[]
        
        for transition in mini_batch:
            s, a, r, s_prime, done_mask = transition
            s_lst.append(s)
            a_lst.append([a])
            r_lst.append([r])
            s_prime_lst.append(s_prime)
            done_mask_lst.append([done_mask])
            
        return torch.tensor(s_lst, dtype=torch. float),torch.tensor(a_lst), torch.tensor(r_lst), torch.tensor(s_prime_lst, dtype=torch. float), torch.tensor(done_mask_lst)
    
    def size(self):
        return len(self.buffer)

class Policy(nn.Module):
    def __init__(self):
        super(Policy, self).__init__()
        self.data = []
        self.fc1 = nn.Linear(16,64)
        self.fc2 = nn.Linear(64,32)
        self.fc3 = nn.Linear(32,6)
        self.number_of_time_list = np.array([1 for x in range(6)])
        self.optimizer = optim.Adam(self.parameters(), lr=learning_rate)
        
    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x
    
    def put_data(self, item):
        self.data.append(item)
    
    def train_net(self):
        R=0
        self.optimizer.zero_grad()
        for r, prob in self.data[::-1]:
            R = r + gamma * R
            loss = -R * torch.log(prob)
            loss.backward()
        self.optimizer.step()
        self.data= []
            
def main():
    env = FJSP_simulator('C:/Users/parkh/FJSP_SIM4.csv','C:/Users/parkh/FJSP_SETUP_SIM.csv',1)
    pi = Policy()
    memory = ReplayBuffer()
    print_interval = 1
    q_load = 20
    score = 0.0
    
    for n_epi in range(1000):
        s = env.reset()
        done = False
        score = 0.0
        while not done:
            prob = pi(torch.from_numpy(s). float())
            m = Categorical(prob)
            a = m.sample()
            s_prime, r, done2 = env.step(a.item())
            z=0
            if done2 == False:
                z+=1
                done_mask =0.0 if done else 1.0
                pi.put_data((r,prob[a]))
                s = s_prime
                score += r
            else:
                done = env.process_event()
            if done:
                break
        pi.train_net()    
        if n_epi % print_interval==0 and n_epi!=0:
            #q_target.load_state_dict(q.state_dict())
            Flow_time, machine_utill ,util, makespan = env.performance_measure()
            print("--------------------------------------------------")
            print("flow time: {}, util : {:.3f}, makespan : {}".format(Flow_time, util, makespan))
            print("n_episode: {}, score : {:.1f}, n_buffer : {}, eps : {:.1f}%".format(n_epi, score/print_interval,memory.size(),epsilon*100))
            #score=0.0
        s = env.reset()
        done = False
        score = 0.0
    while not done:
        a,out = q.select_action(torch.from_numpy(s). float(), epsilon)
        s_prime, r, done2 = env.step(a)
        z=0
        if done2 == False:
            z+=1
            s = s_prime
            print(out)
            print(a)
            score += r
        else:
            done = env.process_event()
        if done:
            break
    Flow_time, machine_util, util, makespan = env.performance_measure()
    print(z)
    return Flow_time, machine_util, util, makespan, score
Flow_time, machine_util, util, makespan, score =main()
print("FlowTime:" , Flow_time)
print("machine_util:" , machine_util)
print("util:" , util)
print("makespan:" , makespan)
print("Score" , score)
      
    