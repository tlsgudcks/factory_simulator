# -*- coding: utf-8 -*-
"""
Created on Mon Apr  4 23:28:32 2022

@author: parkh
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

import gym
import collections
import random
from SIM4 import *

learning_rate = 0.0005
gamma = 1
buffer_limit = 30000
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

class Qnet(nn.Module):
    def __init__(self):
        super(Qnet, self).__init__()
        self.fc1 = nn.Linear(16,64)
        self.fc2 = nn.Linear(64,32)
        self.fc3 = nn.Linear(32,6)
    
    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x
    
    def sample_action(self, obs, epsilon):
        out = self.forward(obs)
        coin = random.random()
        if coin < epsilon:
            return random.randint(0, 5)
        else:
            return out.argmax().item()
    def select_action(self, obs, epsilon):
        out = self.forward(obs)
        return out.argmax().item(),out
        
def train(q, q_target, memory, optimizer):
    for i in range(10):
        s,a,r,s_prime,done_mask = memory.sample(batch_size)
            
        q_out = q(s)
        q_a = q_out.gather(1,a)
        max_q_prime = q_target(s_prime).max (1)[0].unsqueeze(1)
        target = r + gamma * max_q_prime * done_mask
        loss = F.smooth_l1_loss(q_a, target)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
            
def main():
    env = FJSP_simulator('C:/Users/parkh/FJSP_SIM4.csv','C:/Users/parkh/FJSP_SETUP_SIM.csv',1)
    q = Qnet()
    q_target = Qnet()
    q_target.load_state_dict(q.state_dict())
    memory = ReplayBuffer()
    print_interval = 20
    score = 0.0
    optimizer = optim.Adam(q.parameters(), lr=learning_rate)
    
    for n_epi in range(1000):
        epsilon = max(0.01 , 0.08 - 0.01*(n_epi/200))
        s = env.reset()
        done = False
        
        while not done:
            a = q.sample_action(torch.from_numpy(s). float(), epsilon)
            s_prime, r, done2 = env.step(a)
            z=0
            if done2 == False:
                z+=1
                done_mask =0.0 if done else 1.0
                memory.put((s,a,r,s_prime,done_mask))
                s = s_prime
                score += r
            else:
                done = env.process_event()
            if done:
                break
        if memory.size()>2000:
            train(q, q_target, memory, optimizer)
            
        if n_epi % print_interval==0 and n_epi!=0:
            q_target.load_state_dict(q.state_dict())
            print("n_episode: {}, score : {:.1f}, n_buffer : {}, eps : {:.1f}%".format(n_epi, score/print_interval,memory.size(),epsilon*100))
            score=0.0
        s = env.reset()
        done = False
        
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
    return Flow_time, machine_util, util, makespan
Flow_time, machine_util, util, makespan=main()
print("FlowTime:" , Flow_time)
print("machine_util:" , machine_util)
print("util:" , util)
print("makespan:" , makespan)
      
    