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
from Resource2 import *
from Job2 import *
from collections import defaultdict
    
    
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
        s=[]
        """job 인스턴스 생성"""
        self.j_list = defaultdict(Job)
        for i in range(self.job_number):
            j = Job(i+1, self.max_operation[i],self.setup_time_table) 
            self.j_list[j.id] = j
            s.append(j.remain_operation)

        """machine 인스턴스 생성"""
        
        self.r_list = defaultdict(Resource)
        for i in range(self.machine_number):
            r = Resource("M"+str(i+1))
            self.r_list[r.id] = r
            s.append(r.reservation_time)
        for machine in self.r_list:
            s.append(self.r_list[machine].setup_status)
        for job in self.j_list:
            max_op = self.j_list[job].max_operation
            for i in range(max_op):
                s.append(self.j_list[job].operation_in_machine[i])
        self.time = 0 #시간
        self.end = True #종료조건
        self.j = 0
        self.plotlydf = pd.DataFrame([],columns=['Task','Start','Finish','Resource'])
        #remain operation ,last work finish time
        df = pd.Series(s)
        s = df.to_numpy()
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
        s_prime = self.set_state()
        max_reservation_time = 0
        for machine in self.r_list:
            if max_reservation_time < self.r_list[machine].reservation_time:
                max_reservation_time = self.r_list[machine].reservation_time
        r =  self.time - max_reservation_time
        return s_prime, r , done2
    
    def set_state(self):
        s=[]
        for job in self.j_list:
            s.append(self.j_list[job].remain_operation)
        for machine in self.r_list:
            s.append(self.r_list[machine].reservation_time)
        for machine in self.r_list:
            s.append(self.r_list[machine].setup_status)
        for job in self.j_list:
            max_op = self.j_list[job].max_operation
            for i in range(max_op):
                s.append(self.j_list[job].operation_in_machine[i])                
        df = pd.Series(s)
        s = df.to_numpy()
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
        if self.num_of_op == 0:
            done = True
        else:
            done = False
        return done
    
    def assign_setting(self, job, machine, reservation_time): #job = 1 machine = 1
        job.assign_setting(machine)
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
