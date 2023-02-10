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

learning_rate = 0.001  
gamma = 1
buffer_limit = 10000
batch_size = 16
n_rollout = 10
class ActorCritic(nn.Module):
    def __init__(self):
        super(ActorCritic, self).__init__()
        
        self.data = []
        
        self.fc1 = nn.Linear(16,64)
        self.fc2 = nn.Linear(64,32)
        self.fc_pi
        self.fc3 = nn.Linear(32,6)
        self.number_of_time_list = np.array([1 for x in range(6)])
        self.optimizer = optim.Adam(self.parameters(), lr=learning_rate)
        
    def pi(self, x, softmax_dim = 0):
        x = F.relu(self.fc1(x))
        x = self.fc_pi(x)
        prob = F.softmax(x, dim=softmax_dim)
        return prob
    
    def v(self, x):
        x = F.relu(self.fc1(x))
        v = self.fc_v(x)
        return v
    
    def make_batch(self):
        s_lst, a_lst, r_lst, s_prime_lst, done_lst = [],[],[],[],[]
        for transition in self.data:
            s, a, r, s_prime, done = transition
            s_lst.append(s)
            a_lst.append([a])
            r_lst.append([r])
            s_prime_lst.append(s_prime)
            done_mask = 0.0 if done else 1.0
            done_lst.append([done_mask])
            
        s_batch, a_batch, r_batch, s_prime_batch, done_batch = torch.tensor(s_lst, dtype=torch. float), torch.tensor(a_lst), torch.tensor(r_lst, dtype=torch. float), torch.tensor(s_prime_lst, dtype=torch. float), torch.tensor(done_lst, dtype=torch. float)
        self.data = []
        return s_batch, a_batch, r_batch, s_prime_batch, done_batch
    def train_net(self):
        s,a,r,s_prime,done = self.make_batch()
        td_target = r + gamma * self.v(s_prime) * done
        delta = td_target - self.v(s)
    
        pi = self.pi(s, softmax_dim=1)
        pi_a = pi.gather(1,a)
        loss = -torch.log(pi_a) * delta.detach() + F.smooth_l1_loss(self.v(s), td_target.detach())
    
        self.optimizer.zero_grad()
        loss.mean().backward()
        self.optimizer.step()
            
def main():
    env = FJSP_simulator('C:/Users/parkh/FJSP_SIM4.csv','C:/Users/parkh/FJSP_SETUP_SIM.csv',1)
    model = ActorCritic()
    print_interval = 1
    q_load = 20
    score = 0.0
    
    for n_epi in range(1000):
        s = env.reset()
        done = False
        score = 0.0
        while not done:
            for t in range(n_rollout):
                prob = model.pi(torch.from_numpy(s).float())
                m = Categorical(prob)
                a = sample().item()
                s_prime, r, done2 = env.step(a)
                if done2 == False:
                    z+=1
                    done_mask =0.0 if done else 1.0
                    model.put((s,a,r,s_prime,done_mask))
                    
                    s = s_prime
                    score += r
                else:
                    done = env.process_event()
                if done:
                    break
            model.train_net()
            
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
    Flow_time, machine_util, util, makespan = env.performance_measure()
    print(z)
    return Flow_time, machine_util, util, makespan, score
Flow_time, machine_util, util, makespan, score =main()
print("FlowTime:" , Flow_time)
print("machine_util:" , machine_util)
print("util:" , util)
print("makespan:" , makespan)
print("Score" , score)
      
    