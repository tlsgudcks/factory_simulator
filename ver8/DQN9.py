import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

import gym
import collections
import random
from FAB2 import *

learning_rate = 0.001  
gamma = 1
buffer_limit = 10000
batch_size = 16

class Qnet(nn.Module):
    def __init__(self):
        super(Qnet, self).__init__()
        self.fc1 = nn.Linear(64,64)
        self.fc2 = nn.Linear(64,32)
        self.fc3 = nn.Linear(32,10)
    
    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x
    
    def sample_action(self, obs, epsilon):
        out = self.forward(obs)
        coin = random.random()
        if coin < epsilon:
            return random.randint(0, 9)
        else:
            return out.argmax().item()
        
    def select_action(self, obs, epsilon):
        out = self.forward(obs)
        return out.argmax().item(),out
        
def train(q, q_target, memory, optimizer):
    for i in range(10):
        s,a,r,s_prime,done_mask = memory.sample(batch_size)
        #q.number_of_time_list[a] += 1    
        q_out = q(s)
        q_a = q_out.gather(1,a)
        max_q_prime = q_target(s_prime).max (1)[0].unsqueeze(1)
        target = r + gamma * max_q_prime * done_mask
        loss = F.smooth_l1_loss(q_a, target)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

params = torch.load("221q.pt")
q = Qnet()
q.load_state_dict(params)
q.eval()
env = FJSP_simulator('C:/Users/parkh/git_tlsgudcks/simulator/data/FJSP_SIM7_all.csv','C:/Users/parkh/FJSP_SETUP_SIM.csv',"C:/Users/parkh/git_tlsgudcks/simulator/data/FJSP_Fab8.csv",1) 
s = env.reset()
done = False
score = 0.0
epsilon = max(0.01 , 0.08 - 0.02*(20/200))
while not done:
    a, a_list = q.select_action(torch.from_numpy(s). float(), epsilon)
    #print(a_list)
    #print(a)
    s_prime, r, done = env.step(a)
    #print(r)
    s = s_prime
    score += r
    if done:
        break
Flow_time, machine_util, util, makespan = env.performance_measure()

env.gannt_chart()
print("FlowTime:" , Flow_time)
print("machine_util:" , machine_util)
print("util:" , util)
print("makespan:" , makespan)
print("Score" , score)    