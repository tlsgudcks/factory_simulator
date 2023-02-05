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


class FJSP_simulator():
    
    def __init__(self, data):
        self.process_time_table = pd.read_csv(data,index_col=(0))
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
        
    
    def reset(self):
        self.machine_state = np.array([0 for x in range(len(self.process_time_table.columns))])
        self.job_state = [0 for x in range(self.job_number)]
        self.job_operation = [1 for x in range(self.job_number)]
    
    def printer(self):
        print(self.process_time_table)
        print(self.machine_state)
        print(self.job_state)
        print(self.max_operation)
        print(self.time)
        print(self.number_of_operation)
        print(self.unload_operation)
        assign_jobs = self.MOX_SPT(self.process_time_table, self.job_state, 
                                                self.machine_state, self.max_operation,
                                                self.job_operation)
        
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
    
    def run(self):
        plotlydf = pd.DataFrame([],columns=['Task','Start','Finish','Resource']) #간트차트로 보여주기 위한 데이터프레임
        assign_seq = [] #할당 순서를 기록하기 위한 리스트인데 필요는 없음
        j=0 #간트차트 용 인덱스
        while self.end:
            while 0 in self.machine_state:
                assign_jobs = self.SPT_operator(self.process_time_table, self.job_state, 
                                                self.machine_state, self.max_operation,
                                                self.job_operation)
                if assign_jobs[0]==0:
                    break
                self.job_state[assign_jobs[0]-1] = assign_jobs[2]
                self.job_operation[assign_jobs[0]-1] += 1
                self.machine_state[assign_jobs[2]-1] +=assign_jobs[1]
                assign_seq.append(assign_jobs)
                
                
                op = self.job_operation[assign_jobs[0]-1]
                jop = 'j'+str(assign_jobs[0])+str(op-1)
                self.unload_operation.remove(jop)
                self.load_operation.append(jop)
                time = self.time
                start = datetime.fromtimestamp(time*3600)
                time = time+assign_jobs[1]
                end = datetime.fromtimestamp(time*3600)
                job = assign_jobs[0]
                machine = "M"+str(assign_jobs[2])
                plotlydf.loc[j] = dict(Task=job, Start=start, Finish=end, Resource=machine) #간트차트를 위한 딕셔너리 생성, 데이터프레임에 집어넣음
                j+=1
            if self.time%10 == 0:
                self.factory()
            self.time+=1
            self.machine_state -=1
            for i in range(len(self.machine_state)):
                self.machine_state[i] = max(0, self.machine_state[i])
            for i in range(self.job_number):
                if self.machine_state[self.job_state[i]-1] == 0 and self.job_state[i] != 0:
                    self.job_state[i] = 0
                    job=i+1
                    op = self.job_operation[i]
                    jop = 'j'+str(job)+str(op-1)
                    self.load_operation.remove(jop)
                    self.complete_operation.append(jop)
            if len(assign_seq) == self.number_of_operation and self.machine_state.sum() == 0:
                self.end = False
            
        print(assign_seq)    
        print(plotlydf)
        fig = px.timeline(plotlydf, x_start="Start", x_end="Finish", y="Resource", color="Task", width=1000, height=400)
        fig.show()
    
    def MOR_SPT(self, process_time_table, job_state, machine_state, max_operation, job_operation):
        assign_jobs = [0,0,0]
        mor_table=[]
        mor_table2 = []
        for i in range(len(max_operation)):
            mor_table.append(max_operation[i] - job_operation[i])
        for j in range(len(mor_table)):
            if mor_table[j] == max(mor_table):
                mor_table2.append(j)
        for max_index in mor_table2:
            if max_index < 9:
                job = "0"+str(max_index+1)
            else:
                job = str(max_index+1)
            jop = 'j'+ job +"0"+str(job_operation[max_index])
            if job_state[max_index] != 0 :
                assign_jobs=[0,0,0]
            elif jop not in process_time_table.index:
                assign_jobs = [0,0,0]
            else:
                p_table = copy.deepcopy(process_time_table.loc[jop])
                for i in range(len(p_table)):
                    if p_table[i] == 0:
                        p_table[i] = 999
                    if machine_state[i] != 0:
                        p_table[i] = 999
                print(p_table)
                if min(p_table) == 999:
                    assign_jobs =[0,0,0]
                else:
                    p_table = list(p_table)
                    min_index = p_table.index(min(p_table))
                    assign_jobs = [max_index+1,min(p_table),min_index+1]
                    break
        return assign_jobs
    def SPT_operator(self, process_time_table, job_state, machine_state, max_operation, job_operation):
        assign_jobs=[0,0,0]
        for i in range(len(machine_state)):
            if machine_state[i] == 0:
                machine = 'M'+str(i+1)
                p_table = []
                for j in range(self.job_number):
                    if j < 9:
                        mor2 = "0"+str(j+1)
                    else:
                        mor2 = str(j+1)
                    jop = 'j'+ mor2 +"0"+str(job_operation[j])
                    if jop not in process_time_table.index:
                        p_table.append(999)
                    elif process_time_table[machine].loc[jop] != 0 :
                        p_table.append(process_time_table[machine].loc[jop])
                    else:
                        p_table.append(999)
                for k in range(self.job_number):
                    if job_state[k] != 0:
                        p_table[k] = 999
                #print(p_table)
                if min(p_table) == 999:
                    break
                else:
                    min_index = p_table.index(min(p_table))
                    assign_jobs = [min_index+1,min(p_table),i+1]
                    break
        return assign_jobs
 #6, 2, 4, 5, 5, 2, 3, 2, 5, 4, 5       
 #6, 5, 4, 5, 6, 5, 3, 7, 5, 6, 5
main = FJSP_simulator('C:/Users/parkh/FJSP_SIM.csv')
#main.printer()
main.run()
#main.printer()