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
    
class FJSP_simulator():
        
    def __init__(self, data, setup_data, k):
        self.k = k
        self.done = False
        self.event_list=[]
        self.process_time_table = pd.read_csv(data,index_col=(0))
        self.setup_time_table = pd.read_csv(setup_data, index_col=(0))
        self.machine_number = len(self.process_time_table.columns)
        """총 job 개수"""
        operation = self.process_time_table.index
        op_table=[]
        for i in range(len(operation)):
            op_table.append(operation[i][1:3])
        self.job_number = len(set(op_table))
        
        """각 job별로 총 operation개수"""
        self.max_operation=[0 for x in range(self.job_number)]
        for i in range(1, self.job_number+1):
            for j in op_table:
                if i == int(j):
                    self.max_operation[i-1] +=1
        self.num_of_op = sum(self.max_operation)
        """job 인스턴스 생성"""
        self.j_list = defaultdict(Job)
        for i in range(self.job_number):
            j = Job(i+1, self.max_operation[i],self.setup_time_table) 
            self.j_list[j.id] = j

        """machine 인스턴스 생성"""
        self.r_list = defaultdict(Resource)
        for i in range(self.machine_number):
            r = Resource("M"+str(i+1))
            self.r_list[r.id] = r
        self.time = 0 #시간
        self.end = True #종료조건
        self.j=0
        self.plotlydf = pd.DataFrame([],columns=['Task','Start','Finish','Resource'])
        
    def reset(self):
        self.done = False
        self.event_list=[]
        self.machine_number = len(self.process_time_table.columns)
        """총 job 개수"""
        operation = self.process_time_table.index
        op_table=[]
        for i in range(len(operation)):
            op_table.append(operation[i][1:3])
        self.job_number = len(set(op_table))
        
        """각 job별로 총 operation개수"""
        self.max_operation=[0 for x in range(self.job_number)]
        for i in range(1, self.job_number+1):
            for j in op_table:
                if i == int(j):
                    self.max_operation[i-1] +=1
        self.num_of_op = sum(self.max_operation)
        """job 인스턴스 생성"""
        self.j_list = defaultdict(Job)
        for i in range(self.job_number):
            j = Job(i+1, self.max_operation[i],self.setup_time_table) 
            self.j_list[j.id] = j

        """machine 인스턴스 생성"""
        self.r_list = defaultdict(Resource)
        for i in range(self.machine_number):
            r = Resource("M"+str(i+1))
            self.r_list[r.id] = r
        self.time = 0 #시간
        self.end = True #종료조건
        self.j=0
        self.plotlydf = pd.DataFrame([],columns=['Task','Start','Finish','Resource'])
        s="00000000"
        return s
    def performance_measure(self):
        Flow_time = 0
        machine_util = []
        util = 0
        makespan = self.time
        for machine in self.r_list:
            util = self.r_list[machine].util()
            machine_util.append(util)
        util = sum(machine_util)/len(self.r_list)
        for job in self.j_list:
            Flow_time += self.j_list[job].job_flowtime
        return Flow_time, machine_util, util, makespan
    

    #오퍼레이션 길이 50,메이크스팬 100, Max op 5,Min op 5, Max-min
    #39025431
    def step(self, action):
        done2 = False
        a = len(self.event_list)
        self.dispatching_rule_decision(action)
        b = len(self.event_list)
        if a == b:
            done2 = True
        operation_number = 0
        max_op = -1
        min_op = 999
        max_reservation_time = 0
        for machine in self.r_list:
            if max_reservation_time < self.r_list[machine].reservation_time:
                max_reservation_time = self.r_list[machine].reservation_time
        for job in self.j_list:
            remain = self.j_list[job].remain_operation
            if remain <= min_op:
                min_op = remain
            if remain >= max_op:
                max_op = remain
            operation_number += self.j_list[job].remain_operation
        s_prime = self.set_state(operation_number,max_reservation_time,max_op,min_op)
        r = self.time-max_reservation_time
        if self.num_of_op == 1:
            done = True
        else:
            done = False
        return s_prime, r , done, done2
    
    def set_state(self,operation_number, makespan, max_op, min_op):
        if operation_number<10:
            operation_number = "0"+str(operation_number)
        else:
            operation_number = str(operation_number)
        if makespan <10:
            makespan= "00"+str(makespan)
        elif makespan<100:
            makespan= "0"+str(makespan)
        else:
            makespan = str(makespan)
        s = str(max_op)+str(min_op)+str(max_op-min_op)+operation_number+makespan
        return s
    def run(self):
        self.dispatching_rule_decision("random")
        a=0
        while len(self.event_list) != 0:
            self.process_event()
            a+=1
        Flow_time, machine_util, util, makespan = self.performance_measure()
        print("FlowTime:" , Flow_time)
        print("machine_util:" , machine_util)
        print("util:" , util)
        print("makespan:" , makespan)
        print(a)
        fig = px.timeline(self.plotlydf, x_start="Start", x_end="Finish", y="Resource", color="Task", width=1000, height=400)
        fig.show()
    #event = (job_type, operation, machine_type, start_time, end_time, event_type)
    def dispatching_rule_decision(self, a):
        if a == "random":
            coin = random.randint(0,5)
        else:
            coin = int(a)
        if coin == 0:
            self.dispatching_rule_SPT()
        elif coin == 1:
            self.dispatching_rule_SPTSSU()
        elif coin == 2:
            self.dispatching_rule_MOR()
        elif coin == 3:
            self.dispatching_rule_MORSPT()
        elif coin == 4:
            self.dispatching_rule_LOR()
        elif coin == 5:
            self.dispatching_rule_LPT()
        
    def process_event(self):
        self.event_list.sort(key = lambda x:x[3], reverse = False)
        event = self.event_list.pop(0)
        self.time = event[3]
        job = event[0]
        machine = event[1]
        time = event[2]
        start = datetime.fromtimestamp(time*3600)
        time = event[3]
        end = datetime.fromtimestamp(time*3600)
        if event[4] == "setup_change":
            event_type = "setup"
        else:
            event_type = "j"+str(event[0].id)
            job.complete_setting(event[2], event[3],event[4]) # 작업이 대기로 변함, 시작시간, 종료시간, event_type
            machine.complete_setting(event[2],event[3],event[4]) # 기계도 사용가능하도록 변함
            #self.dispatching_rule_decision("random")
            self.num_of_op -=1
        self.plotlydf.loc[self.j] = dict(Task=event_type, Start=start, Finish=end, Resource=machine.id) #간트차트를 위한 딕셔너리 생성, 데이터프레임에 집어넣음
        self.j+=1
    
    def assign_setting(self, job, machine, reservation_time): #job = 1 machine = 1
        job.assign_setting()
        machine.assign_setting(job, reservation_time)
        
    def complete_setting(self, job, machine):
        job.complete_setting()
        machine.complete_setting()
            
    def dispatching_rule_SPT(self):
        for machine in self.r_list:
            if self.r_list[machine].status == 0:
                machine = self.r_list[machine].id #machine 이름
                p_table=[]
                for job in self.j_list: #job 이름과 operation이름 찾기
                    jop = self.j_list[job].jop()
                    if jop not in self.process_time_table.index: #해당 jop가 없는 경우  
                        pass
                    elif self.process_time_table[machine].loc[jop] == 0 : #해당 jop가 작업이 불가능할 경우
                        pass
                    elif self.j_list[job].status != "WAIT": #해당 jop가 작업중일 경우
                        pass
                    else: 
                        p_table.append([self.j_list[job], self.process_time_table[machine].loc[jop]])
                if len(p_table) == 0:#현재 이벤트를 발생시킬 수 없음
                    pass
                else:
                    p_table.sort(key = lambda x:x[1], reverse = False)
                    setup_time = p_table[0][0].setup_table['j'+str(self.r_list[machine].setup_status)] #컬럼에서 machine에 세팅되어있던 job에서 변경유무 확인
                    if setup_time !=0:
                        self.event_list.append((p_table[0][0], self.r_list[machine], self.time, self.time+setup_time,"setup_change"))
                    self.event_list.append((p_table[0][0], self.r_list[machine], self.time+setup_time, self.time+setup_time+p_table[0][1],"track_in_finish"))
                    self.assign_setting(p_table[0][0], self.r_list[machine],self.time+setup_time+p_table[0][1])
                    break
            
    def dispatching_rule_LPT(self):
        
        
        for machine in self.r_list:
            if self.r_list[machine].status == 0:
                machine = self.r_list[machine].id #machine 이름
                p_table=[]
                for job in self.j_list: #job 이름과 operation이름 찾기
                    jop = self.j_list[job].jop()
                    if jop not in self.process_time_table.index: #해당 jop가 없는 경우  
                        pass
                    elif self.process_time_table[machine].loc[jop] == 0 : #해당 jop가 작업이 불가능할 경우
                        pass
                    elif self.j_list[job].status != "WAIT": #해당 jop가 작업중일 경우
                        pass
                    else: 
                        p_table.append([self.j_list[job], self.process_time_table[machine].loc[jop]])
                if len(p_table) == 0:#현재 이벤트를 발생시킬 수 없음
                    pass
                else:
                    p_table.sort(key = lambda x:x[1], reverse = True)
                    setup_time = p_table[0][0].setup_table['j'+str(self.r_list[machine].setup_status)] #컬럼에서 machine에 세팅되어있던 job에서 변경유무 확인
                    if setup_time !=0:
                        self.event_list.append((p_table[0][0], self.r_list[machine], self.time, self.time+setup_time,"setup_change"))
                    self.event_list.append((p_table[0][0], self.r_list[machine], self.time+setup_time, self.time+setup_time+p_table[0][1],"track_in_finish"))
                    self.assign_setting(p_table[0][0], self.r_list[machine],self.time+setup_time+p_table[0][1])
                    break
    def dispatching_rule_SPTSSU(self):
        for machine in self.r_list:
            if self.r_list[machine].status == 0:
                machine = self.r_list[machine].id #machine 이름
                p_table=[]
                for job in self.j_list: #job 이름과 operation이름 찾기
                    jop = self.j_list[job].jop()
                    setup_time = self.j_list[job].setup_table['j'+str(self.r_list[machine].setup_status)]
                    if jop not in self.process_time_table.index: #해당 jop가 없는 경우  
                        pass
                    elif self.process_time_table[machine].loc[jop] == 0 : #해당 jop가 작업이 불가능할 경우
                        pass
                    elif self.j_list[job].status != "WAIT": #해당 jop가 작업중일 경우
                        pass
                    else: 
                        p_table.append([self.j_list[job], self.process_time_table[machine].loc[jop]+setup_time])
                if len(p_table) == 0:#현재 이벤트를 발생시킬 수 없음
                    pass
                else:
                    p_table.sort(key = lambda x:x[1], reverse = False)
                    setup_time = p_table[0][0].setup_table['j'+str(self.r_list[machine].setup_status)] #컬럼에서 machine에 세팅되어있던 job에서 변경유무 확인
                    if setup_time !=0:
                        self.event_list.append((p_table[0][0], self.r_list[machine], self.time, self.time+setup_time,"setup_change"))
                    self.event_list.append((p_table[0][0], self.r_list[machine], self.time+setup_time, self.time+p_table[0][1],"track_in_finish"))
                    self.assign_setting(p_table[0][0], self.r_list[machine],self.time+setup_time+p_table[0][1])
                    break
            
    def dispatching_rule_MOR(self):
        for machine in self.r_list:
            if self.r_list[machine].status == 0:
                machine = self.r_list[machine].id #machine 이름
                p_table=[]
                for job in self.j_list: #job 이름과 operation이름 찾기
                    jop = self.j_list[job].jop()
                    setup_time = self.j_list[job].setup_table['j'+str(self.r_list[machine].setup_status)]
                    if jop not in self.process_time_table.index: #해당 jop가 없는 경우  
                        pass
                    elif self.process_time_table[machine].loc[jop] == 0 : #해당 jop가 작업이 불가능할 경우
                        pass
                    elif self.j_list[job].status != "WAIT": #해당 jop가 작업중일 경우
                        pass
                    else: 
                        p_table.append([self.j_list[job], self.process_time_table[machine].loc[jop]])
                if len(p_table) == 0:#현재 이벤트를 발생시킬 수 없음
                    pass
                else:
                    p_table.sort(key = lambda x:x[0].remain_operation, reverse = True)
                    setup_time = p_table[0][0].setup_table['j'+str(self.r_list[machine].setup_status)] #컬럼에서 machine에 세팅되어있던 job에서 변경유무 확인
                    if setup_time !=0:
                        self.event_list.append((p_table[0][0], self.r_list[machine], self.time, self.time+setup_time,"setup_change"))
                    self.event_list.append((p_table[0][0], self.r_list[machine], self.time+setup_time, self.time+setup_time+p_table[0][1],"track_in_finish"))
                    self.assign_setting(p_table[0][0], self.r_list[machine],self.time+setup_time+p_table[0][1])
                    break
            
    def dispatching_rule_MORSPT(self):
        for machine in self.r_list:
            if self.r_list[machine].status == 0:
                machine = self.r_list[machine].id #machine 이름
                p_table=[]
                for job in self.j_list: #job 이름과 operation이름 찾기
                    jop = self.j_list[job].jop()
                    if jop not in self.process_time_table.index: #해당 jop가 없는 경우  
                        pass
                    elif self.process_time_table[machine].loc[jop] == 0 : #해당 jop가 작업이 불가능할 경우
                        pass
                    elif self.j_list[job].status != "WAIT": #해당 jop가 작업중일 경우
                        pass
                    else: 
                        p_table.append([self.j_list[job], self.process_time_table[machine].loc[jop]])
                if len(p_table) == 0:#현재 이벤트를 발생시킬 수 없음
                    pass
                else:
                    p_table.sort(key = lambda x : x[1]/x[0].remain_operation, reverse = False)
                    setup_time = p_table[0][0].setup_table['j'+str(self.r_list[machine].setup_status)] #컬럼에서 machine에 세팅되어있던 job에서 변경유무 확인
                    if setup_time !=0:
                        self.event_list.append((p_table[0][0], self.r_list[machine], self.time, self.time+setup_time,"setup_change"))
                    self.event_list.append((p_table[0][0], self.r_list[machine], self.time+setup_time, self.time+p_table[0][1]+setup_time,"track_in_finish"))
                    self.assign_setting(p_table[0][0], self.r_list[machine],self.time+setup_time+p_table[0][1])
                    break
            
    def dispatching_rule_LOR(self):
        for machine in self.r_list:
            if self.r_list[machine].status == 0:
                machine = self.r_list[machine].id #machine 이름
                p_table=[]
                for job in self.j_list: #job 이름과 operation이름 찾기
                    jop = self.j_list[job].jop()
                    setup_time = self.j_list[job].setup_table['j'+str(self.r_list[machine].setup_status)]
                    if jop not in self.process_time_table.index: #해당 jop가 없는 경우  
                        pass
                    elif self.process_time_table[machine].loc[jop] == 0 : #해당 jop가 작업이 불가능할 경우
                        pass
                    elif self.j_list[job].status != "WAIT": #해당 jop가 작업중일 경우
                        pass
                    else: 
                        p_table.append([self.j_list[job], self.process_time_table[machine].loc[jop]])
                if len(p_table) == 0:#현재 이벤트를 발생시킬 수 없음
                    pass
                else:
                    p_table.sort(key = lambda x:x[0].remain_operation, reverse = False)
                    setup_time = p_table[0][0].setup_table['j'+str(self.r_list[machine].setup_status)] #컬럼에서 machine에 세팅되어있던 job에서 변경유무 확인
                    if setup_time !=0:
                        self.event_list.append((p_table[0][0], self.r_list[machine], self.time, self.time+setup_time,"setup_change"))
                    self.event_list.append((p_table[0][0], self.r_list[machine], self.time+setup_time, self.time+setup_time+p_table[0][1],"track_in_finish"))
                    self.assign_setting(p_table[0][0], self.r_list[machine],self.time+setup_time+p_table[0][1])
                    break
#6, 2, 4, 5, 5, 2, 3, 2, 5, 4, 5       
#6, 5, 4, 5, 6, 5, 3, 7, 5, 6, 5
class QAgent():
    def __init__(self):
        self.q_table = np.zeros((100000000,6))
        self.eps = 0.9
        self.alpha = 0.01
    
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
            s_prime, r, done, done2 = env.step(a)
            if done2 == False:
                z+=1
            if done2 == False:
                agent.update_table((s,a,r,s_prime))
                s = s_prime
            else:
                env.process_event()
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
        s_prime, r, done,done2 = env.step(a)
        if done2 == True:
            env.process_event()
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