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
import Resource
import Job


class FJSP_simulator():
    
    def __init__(self, data, setup_data):
        self.event_list=[]
        self.process_time_table = pd.read_csv(data,index_col=(0))
        self.setup_time_table = pd.read_csv(setup_data, index_col=(0))
        self.machine_state = np.array([0 for x in range(len(self.process_time_table.columns))])
        operation = self.process_time_table.index
        op_table=[]
        for i in range(len(operation)):
            op_table.append(operation[i][1:3])
        self.job_number = len(set(op_table))
        self.max_operation=[0 for x in range(self.job_number)]
        for i in range(1, self.job_number+1):
            for j in op_table:
                if i == int(j):
                    self.max_operation[i-1] +=1            
        self.job_state = [0 for x in range(self.job_number)]
        self.job_operation = [1 for x in range(self.job_number)]
        self.machine_setup = [0 for x in range(len(self.machine_state))]
        self.time = 0
        self.end = True
        self.number_of_operation=0
        for i in range(len(self.max_operation)):
            self.number_of_operation += self.max_operation[i]
        self.complete_operation=[]
        self.load_operation=[]
        self.unload_operation=[]
        for i in range(len(self.max_operation)):
            for j in range(self.max_operation[i]):
                self.unload_operation.append('j'+str(i+1)+str(j+1))
        self.j=0
        self.plotlydf = pd.DataFrame([],columns=['Task','Start','Finish','Resource'])
    def reset(self):
        self.machine_state = np.array([0 for x in range(len(self.process_time_table.columns))])
        self.job_state = [0 for x in range(self.job_number)]
        self.job_operation = [1 for x in range(self.job_number)]
    
    def printer(self):
        print(self.machine_state)
        print(self.job_state)
        
    def factory(self):
        print("----현재시각",self.time,"-----------------------------")
        for i in range(len(self.machine_state)):
            if self.machine_state[i] != 0:
                job = self.job_state.index(i+1)
                job_op = self.job_operation[job]
                jop = "j"+str(job+1)+str(job_op-1)
            else:
                jop=""
            k = """ ---M {0}---
 [ {1} {2} ]
 ----------"""
            #print(" ---","M",(i+1),"---","\n","[ ",jop,self.machine_state[i]," ]","\n","-----------")
            print(k.format(i+1,jop,self.machine_state[i]))
        print(self.job_state)
        print(self.max_operation)
        print(self.job_operation)
        print("unload operation: ",self.unload_operation)
        print("loading operation: ", self.load_operation)
        print("complete_operation: ", self.complete_operation)
    def get_state(self):
        state = [self.time, self.job_state, self.machine_state]
        return state
    def initialize(self):
        print("a")
    def run(self):
        self.dispatching_rule_decision("random")
        a=0
        while len(self.event_list) != 0:
            self.process_event()
            a+=1
        print(self.time,a)
        fig = px.timeline(self.plotlydf, x_start="Start", x_end="Finish", y="Resource", color="Task", width=1000, height=400)
        fig.show()
    #event = (job_type, operation, machine_type, start_time, end_time, event_type)
    def dispatching_rule_decision(self, a):
        if a == "random":
            coin = random.randint(0,3)
        else:
            coin = int(a)
        if coin == 0:
            self.dispatching_rule_SPT()
        elif coin == 1:
            self.dispatching_rule_SPTSSU()
        elif coin == 2:
            self.dispatching_rule_MOR()
        elif coin == 3:
            self.dispatching_rule_MORSPT
    
    def process_event(self):
        event = self.event_list.pop(0)
        self.time = event[4]
        machine = event[2]
        time = event[3]
        start = datetime.fromtimestamp(time*3600)
        time = event[4]
        end = datetime.fromtimestamp(time*3600)
        if event[5] == "setup_change":
            job = "setup"
        else:    
            job = event[0]
            self.job_state[event[0]-1] = 0
            self.machine_state[int(event[2][-1])-1] = 0
            self.dispatching_rule_decision("random")
        self.plotlydf.loc[self.j] = dict(Task=job, Start=start, Finish=end, Resource=machine) #간트차트를 위한 딕셔너리 생성, 데이터프레임에 집어넣음
        self.j+=1
    
    def setting(self, job, machine): #job = 1 machine = 1
        self.job_state[job-1] = machine
        self.job_operation[job-1] += 1
        self.machine_state[machine-1] = job
        self.machine_setup[machine-1] = job
    
    
    def dispatching_rule_SPT(self):
        for i in range(len(self.machine_state)):
            if self.machine_state[i] == 0:
                machine = 'M'+str(i+1) #machine 이름
                p_table = []
                for j in range(self.job_number): #job 이름과 operation이름 찾기
                    if j < 9:
                        mor2 = "0"+str(j+1)
                    else:
                        mor2 = str(j+1)
                    jop = 'j'+ mor2 +"0"+str(self.job_operation[j]) #j0101의 형태로 나타남
                    print(jop)
                    print(type(jop))
                    if jop not in self.process_time_table.index: #해당 jop가 없는 경우
                        p_table.append(999)
                    elif self.process_time_table[machine].loc[jop] == 0 :
                        p_table.append(999)
                    elif self.job_state[j] != 0: #해당 jop가 작업중일 경우
                        p_table.append(999)
                    else: #해당 jop가 작업이 불가능할 경우
                        p_table.append(self.process_time_table[machine].loc[jop])
                if min(p_table) == 999:#현재 이벤트를 발생시킬 수 없음
                    break
                else:
                    min_index = p_table.index(min(p_table))
                    df2_sorted = self.setup_time_table['j'+str(min_index+1)] #셋업테이블에서 job에 해당하는 컬럼을 가져옴
                    setup_time = df2_sorted.loc['j'+ str(self.machine_setup[i])] #컬럼에서 machine에 세팅되어있던 job에서 변경유무 확인
                    if setup_time !=0:
                        self.event_list.append((min_index+1,self.job_operation[min_index],machine, self.time, self.time+setup_time,"setup_change"))
                    self.event_list.append((min_index+1,self.job_operation[min_index],machine, self.time+setup_time, self.time+setup_time+p_table[min_index],"track_in_finish"))
                    self.setting(self.event_list[-1][0], i+1)
                    self.printer()
        self.event_list.sort(key = lambda x:x[4], reverse = False)
        
    def dispatching_rule_SPTSSU(self):
        for i in range(len(self.machine_state)):
            if self.machine_state[i] == 0:
                machine = 'M'+str(i+1) #machine 이름
                p_table = []
                for j in range(self.job_number): #job 이름과 operation이름 찾기
                    if j < 9:
                        mor2 = "0"+str(j+1)
                    else:
                        mor2 = str(j+1)
                    jop = 'j'+ mor2 +"0"+str(self.job_operation[j]) #j11의 형태로 나타남
                    df2_sorted = self.setup_time_table["j"+str(j+1)] #셋업테이블에서 job에 해당하는 컬럼을 가져옴
                    setup_time = df2_sorted.loc['j'+ str(self.machine_setup[i])] #컬럼에서 machine에 세팅되어있던 job에서 변경유무 확인
                    print(setup_time)
                    if jop not in self.process_time_table.index: #해당 jop가 없는 경우
                        p_table.append(999)
                    elif self.process_time_table[machine].loc[jop] == 0 :
                        p_table.append(999)
                    elif self.job_state[j] != 0: #해당 jop가 작업중일 경우
                        p_table.append(999)
                    else: #해당 jop가 작업이 불가능할 경우
                        p_table.append(self.process_time_table[machine].loc[jop]+setup_time)
                if min(p_table) == 999:#현재 이벤트를 발생시킬 수 없음
                    break
                else:
                    min_index = p_table.index(min(p_table))
                    df2_sorted = self.setup_time_table['j'+str(min_index+1)] #셋업테이블에서 job에 해당하는 컬럼을 가져옴
                    setup_time = df2_sorted.loc['j'+ str(self.machine_setup[i])] #컬럼에서 machine에 세팅되어있던 job에서 변경유무 확인
                    if setup_time !=0:
                        self.event_list.append((min_index+1,self.job_operation[min_index],machine, self.time, self.time+setup_time,"setup_change"))
                    self.event_list.append((min_index+1,self.job_operation[min_index],machine, self.time+setup_time, self.time+p_table[min_index],"track_in_finish"))
                    self.setting(self.event_list[-1][0], i+1)
                    self.printer()
        self.event_list.sort(key = lambda x:x[4], reverse = False)
        
    def dispatching_rule_MORSPT(self):
        for i in range(len(self.machine_state)):
            if self.machine_state[i] == 0:
                machine = 'M'+str(i+1) #machine 이름
                p_table = []
                mor_table=[]
                for j in range(self.job_number): #job 이름과 operation이름 찾기
                    if j < 9:
                        mor2 = "0"+str(j+1)
                    else:
                        mor2 = str(j+1)
                    jop = 'j'+ mor2 +"0"+str(self.job_operation[j]) #j11의 형태로 나타남
                    df2_sorted = self.setup_time_table["j"+str(j+1)] #셋업테이블에서 job에 해당하는 컬럼을 가져옴
                    setup_time = df2_sorted.loc['j'+ str(self.machine_setup[i])] #컬럼에서 machine에 세팅되어있던 job에서 변경유무 확인
                    if jop not in self.process_time_table.index: #해당 jop가 없는 경우
                        p_table.append(999)
                        mor_table.append(0)
                    elif self.process_time_table[machine].loc[jop] == 0 :
                        p_table.append(999)
                        mor_table.append(0)
                    elif self.job_state[j] != 0: #해당 jop가 작업중일 경우
                        p_table.append(999)
                        mor_table.append(0)
                    else: #해당 jop가 작업이 불가능할 경우
                        mor_table.append(self.max_operation[j]-self.job_operation[j]+1)
                        p_table.append(self.process_time_table[machine].loc[jop])
                if min(p_table) == 999:#현재 이벤트를 발생시킬 수 없음
                    break
                else:
                    mor_p_table=[]
                    for k in range(self.job_number):
                        if mor_table[k] != 0 and p_table[k] != 999:
                            mor_p_table.append(p_table[k]/mor_table[k])
                        else:
                            mor_p_table.append(999)
                    if min(mor_p_table) == 999:#현재 이벤트를 발생시킬 수 없음
                        break
                    print(mor_p_table)
                    min_index = mor_p_table.index(min(mor_p_table))
                    df2_sorted = self.setup_time_table['j'+str(min_index+1)] #셋업테이블에서 job에 해당하는 컬럼을 가져옴
                    setup_time = df2_sorted.loc['j'+ str(self.machine_setup[i])] #컬럼에서 machine에 세팅되어있던 job에서 변경유무 확인
                    if setup_time !=0:
                        self.event_list.append((min_index+1,self.job_operation[min_index],machine, self.time, self.time+setup_time,"setup_change"))
                    self.event_list.append((min_index+1,self.job_operation[min_index],machine, self.time+setup_time, self.time+setup_time+p_table[min_index],"track_in_finish"))
                    self.setting(self.event_list[-1][0], i+1)
                    self.printer()
        self.event_list.sort(key = lambda x:x[4], reverse = False)
        
    def dispatching_rule_MOR(self):
        for i in range(len(self.machine_state)):
            if self.machine_state[i] == 0:
                machine = 'M'+str(i+1) #machine 이름
                p_table = []
                mor_table = []
                for j in range(self.job_number): #job 이름과 operation이름 찾기
                    if j < 9:
                        mor2 = "0"+str(j+1)
                    else:
                        mor2 = str(j+1)
                    jop = 'j'+ mor2 +"0"+str(self.job_operation[j]) #j0101의 형태로 나타남
                    if jop not in self.process_time_table.index: #해당 jop가 없는 경우
                        mor_table.append(0)    
                        p_table.append(999)
                    elif self.process_time_table[machine].loc[jop] == 0 :
                        mor_table.append(0)
                        p_table.append(999)
                    elif self.job_state[j] != 0: #해당 jop가 작업중일 경우
                        mor_table.append(0)
                        p_table.append(999)
                    else: #해당 jop가 작업이 불가능할 경우
                        mor_table.append(self.max_operation[j]-self.job_operation[j])
                        p_table.append(self.process_time_table[machine].loc[jop])
                if min(p_table) == 999:#현재 이벤트를 발생시킬 수 없음
                    break
                else:
                    max_index = -1
                    max_op = -1
                    for j in range(self.job_number):
                        if p_table[j] !=999 and mor_table[j] > max_op:
                            max_op = mor_table[j]
                            max_index = j
                    if max_index == -1:
                        continue
                    else:
                        df2_sorted = self.setup_time_table['j'+str(max_index+1)] #셋업테이블에서 job에 해당하는 컬럼을 가져옴
                        setup_time = df2_sorted.loc['j'+ str(self.machine_setup[i])] #컬럼에서 machine에 세팅되어있던 job에서 변경유무 확인
                        if setup_time !=0:
                            self.event_list.append((max_index+1,self.job_operation[max_index],machine, self.time, self.time+setup_time,"setup_change"))
                        self.event_list.append((max_index+1,self.job_operation[max_index],machine, self.time+setup_time, self.time+setup_time+p_table[max_index],"track_in_finish"))
                        self.setting(self.event_list[-1][0], i+1)
                        self.printer()
        self.event_list.sort(key = lambda x:x[4], reverse = False)
 #6, 2, 4, 5, 5, 2, 3, 2, 5, 4, 5       
 #6, 5, 4, 5, 6, 5, 3, 7, 5, 6, 5
main = FJSP_simulator('C:/Users/parkh/FJSP_SIM.csv','C:/Users/parkh/FJSP_SETUP_SIM.csv')
#main.printer()
main.run()
#main.printer()