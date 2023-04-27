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
        self.plotlydf = pd.DataFrame([],columns=['Type','Task','Start','Finish','Resource','Rule','Step'])
        self.step_number = 1
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
        #s.append(1)
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
        self.plotlydf = pd.DataFrame([],columns=["Type",'Task','Start','Finish','Resource','Rule','Step'])
        #remain operation ,last work finish time
        df = pd.Series(s)
        s = df.to_numpy()
        self.step_number = 1
        return s
    def performance_measure(self):
        Flow_time = 0
        value_time_table = []
        full_time_table = []
        machine_util = 0
        util = 0
        makespan = self.time
        for machine in self.r_list:
            value_added_time, full_time = self.r_list[machine].util()
            value_time_table.append(value_added_time)
            full_time_table.append(full_time)
        util = sum(value_time_table)/sum(full_time_table)
        for job in self.j_list:
            Flow_time += self.j_list[job].job_flowtime
        #fig = px.timeline(self.plotlydf, x_start="Start", x_end="Finish", y="Resource", color="Task", width=1000, height=400)
        #fig.show()
        return Flow_time, machine_util, util, makespan
    def modify_width(self, bar, width):
        """
        막대의 너비를 설정합니다.
        width = (단위 px)
        """
        bar.width = width
    def modify_text(self, bar):
        """
        막대의 텍스트를 설정합니다.
        width = (단위 px)
        """
        bar.text = ""
    def to_bottom_setup_df(self, df):
        """
        figure의 경우 위에서 부터 bar 생성됩니다.
        track_in event를 df(데이터프레임) 가장 밑 행으로 배치시킵니다.
        이 작업을 통해 TRACK_IN 이벤트가 다른 중복되는 차트에 가려지는 것을 방지합니다.
        """
        setup_df = df.loc[df['Type'] == 'setup']
        df = df[df['Type'] != 'setup']
        df = df.append(setup_df, ignore_index=True)
        return df
    
    def gannt_chart(self):
        step_rule = []
        for i in range(len(self.plotlydf)):
            if str(self.plotlydf["Rule"].loc[i])  != "None":
                step_rule.append(str(self.plotlydf["Step"].loc[i])+"-"+str(self.plotlydf["Rule"].loc[i]))
            else:
                step_rule.append("NONE")
        self.plotlydf["Step-Rule"] = step_rule
        
        plotlydf2 = self.plotlydf.sort_values(by=['Resource','Type'], ascending=False)
        df = self.to_bottom_setup_df(plotlydf2) #setup 뒤로 보낸 데이터 프레임
        
        fig = px.timeline(df, x_start="Start", x_end="Finish", y="Resource", hover_data=['Rule'],template="simple_white",color="Type", color_discrete_sequence=px.colors.qualitative.Dark24 ,text = "Task", width=2000, height=800)
        fig.update_traces(marker=dict(line_color="black"))
        
        [(self.modify_width(bar, 0.7))
        for bar in fig.data if ('setup' in bar.legendgroup)]
        fig.show()
        
        #fig,write_html(f"{PathInfo.xlsx}{os.sep}temp_target.html", default_width=2300, default_height=900)
        plotlydf3 = self.plotlydf.sort_values(by=['Type'], ascending=True)
        fig2 = px.timeline(plotlydf3, x_start="Start", x_end="Finish", y="Type", template="seaborn" ,color="Resource",text = "Resource", width=2000, height=1000)
        fig2.update_traces(marker=dict(line_color="yellow", cmid = 1000))
        fig2.show()
        
        fig3 = px.timeline(df, x_start="Start", x_end="Finish", y="Resource", template="simple_white",color="Type", color_discrete_sequence=px.colors.qualitative.Dark24 ,text = "Rule", width=2000, height=800)
        [(self.modify_width(bar, 0.7), self.modify_text(bar)) for bar in fig3.data if ('setup' in bar.legendgroup)]
        fig3.show()
        
        
        fig4 = px.timeline(df, x_start="Start", x_end="Finish", y="Resource", template="simple_white",color="Type", color_discrete_sequence=px.colors.qualitative.Dark24 ,text = "Step-Rule", width=2000, height=800)
        [(self.modify_width(bar, 0.7), self.modify_text(bar))
        for bar in fig4.data if ('setup' in bar.legendgroup)]
        fig4.show()
        
        fig5 = px.timeline(df, x_start="Start", x_end="Finish", y="Rule", template="simple_white",color="Rule", color_discrete_sequence=px.colors.qualitative.Dark24 ,text = "Step-Rule", width=2000, height=800)
        fig5.show()
    #오퍼레이션 길이 50,메이크스팬 100, Max op 5,Min op 5, Max-min
    #39025431
    
    def deliberate_delay_action(self,machine):
        rule_name= "DDA"
        step_num = self.step_number
        machine = self.r_list[machine].id #machine 이름
        self.event_list.sort(key = lambda x:x[4], reverse = False)
        if len(self.event_list) == 0:
            last_time = 0
        else:    
            last_time = self.event_list[0][4]
        self.event_list.append(("delay", "delay", self.r_list[machine], self.time, last_time,"delay_machine",rule_name,step_num,0))
        self.assign_setting("delay", self.r_list[machine], last_time)
        return -(last_time-self.time)
    def step(self, action):
        done = False
        while True:
            process = self.check_process()
            if process == True:
                self.process_event()
                if self.num_of_op == 0:
                    done = True
                    s_prime = self.set_state()
                    r =  0
                    break
            else:
                machine, get_machine = self.check_availability()
                if machine == "NONE":
                    r = self.deliberate_delay_action(get_machine)
                    s_prime = self.set_state()
                    break
                else:
                    p_time = self.dispatching_rule_decision(machine, action)
                    s_prime = self.set_state()
                    reservation_time = self.r_list[machine].reservation_time
                    last_work_finish_time = self.r_list[machine].last_work_finish_time
                    max_reservation = 0
                    min_reservation = 100000000
                    for machine in self.r_list:
                        if self.r_list[machine].reservation_time > max_reservation:
                            max_reservation = self.r_list[machine].reservation_time
                        if self.r_list[machine].reservation_time < min_reservation:
                            min_reservation = self.r_list[machine].reservation_time
                    r = -(reservation_time-last_work_finish_time-p_time)
                    break
        return s_prime, r , done
    
    def set_state(self):
        s=[]
        #machine = self.check_availability()
        #print(machine)
        #s.append(int(machine[1:]))
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
    def dispatching_rule_decision(self,machine, a):
        #print(machine)
        if a == "random":
            coin = random.randint(0,5)
        else:
            coin = int(a)
        if coin == 0:
            p_time = self.dispatching_rule_SPT(machine)
        elif coin == 1:
            p_time = self.dispatching_rule_SPTSSU(machine)
        elif coin == 2:
            p_time = self.dispatching_rule_MOR(machine)
        elif coin == 3:
            p_time = self.dispatching_rule_MORSPT(machine)
        elif coin == 4:
            p_time = self.dispatching_rule_LOR(machine)
        elif coin == 5:
            p_time = self.dispatching_rule_LPT(machine)
        return p_time
    def process_event(self):
        self.event_list.sort(key = lambda x:x[4], reverse = False)
        k = 0
        for i in range(len(self.event_list)):
            if self.event_list[0][4] == self.event_list[i][4]:
                k+=1
        for i in range(k):
            event = self.event_list.pop(0)
            self.time = event[4]
            setup_time = event[8]
            job = event[0]
            machine = event[2]
            time = event[3]
            start = datetime.fromtimestamp(time*3600)
            time = event[4]
            end = datetime.fromtimestamp(time*3600)
            if event[5] == "setup_change":
                event_type = "setup"
            elif event[5] == "delay_machine":
                event_type = "delay_machine"
                machine.complete_setting(event[3]+setup_time,event[4],event[5])
            else:
                event_type = "j"+str(event[0].id)
                job.complete_setting(event[3]+setup_time, event[4],event[5]) # 작업이 대기로 변함, 시작시간, 종료시간, event_type
                machine.complete_setting(event[3]+setup_time,event[4],event[5]) # 기계도 사용가능하도록 변함
                #self.dispatching_rule_decision("random")
                self.num_of_op -=1
            rule = event[6]
            step = event[7]
            operation = event[1]
            self.plotlydf.loc[self.j] = dict(Type = event_type, Task=operation, Start=start, Finish=end, Resource=machine.id, Rule = rule, Step = step) #간트차트를 위한 딕셔너리 생성, 데이터프레임에 집어넣음
            self.j+=1
            if self.num_of_op == 0:
                done = True
            else:
                done = False
        return done
    
    def assign_setting(self, job, machine, reservation_time): #job = 1 machine = 1
        if job != "delay":
            job.assign_setting(machine)
            machine.assign_setting(job, reservation_time)
        else:
            machine.delay_setting(job, reservation_time)
        
    def complete_setting(self, job, machine):
        job.complete_setting()
        machine.complete_setting()
    def check_process(self):
        process = False
        k = 1
        for machine in self.r_list:
            k = k*self.r_list[machine].status
        if k != 0:
            process = True
        return process
    def check_availability(self):
        index_k = 0
        select_machine = "NONE"
        get_machine = ""
        for machine in self.r_list:
            index_k += 1
            if self.r_list[machine].status == 0:
                get_machine = machine
                machine = self.r_list[machine].id #machine 이름
                get_machine = machine
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
                    select_machine = machine
                    break
        return select_machine, get_machine
                
    def dispatching_rule_SPT(self, machine):
        rule_name= "SPT"
        step_num = self.step_number
        self.step_number+=1
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
                p_table.append([self.j_list[job], self.process_time_table[machine].loc[jop],jop])
        if len(p_table) == 0:#현재 이벤트를 발생시킬 수 없음
            pass
        else:
            p_table.sort(key = lambda x:x[1], reverse = False)
            setup_time = p_table[0][0].setup_table['j'+str(self.r_list[machine].setup_status)] #컬럼에서 machine에 세팅되어있던 job에서 변경유무 확인
            jop = p_table[0][2]
            if setup_time !=0:
                self.event_list.append((p_table[0][0],"setup" , self.r_list[machine], self.time, self.time+setup_time,"setup_change","NONE",step_num,setup_time))
            self.event_list.append((p_table[0][0], jop ,self.r_list[machine], self.time, self.time+setup_time+p_table[0][1],"track_in_finish",rule_name,step_num,setup_time))
            self.assign_setting(p_table[0][0], self.r_list[machine],self.time+setup_time+p_table[0][1])
        return p_table[0][1]
    
    def dispatching_rule_LPT(self, machine):
        rule_name= "LPT"
        step_num = self.step_number
        self.step_number+=1
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
                p_table.append([self.j_list[job], self.process_time_table[machine].loc[jop],jop])
        p_table.sort(key = lambda x:x[1], reverse = True)
        setup_time = p_table[0][0].setup_table['j'+str(self.r_list[machine].setup_status)] #컬럼에서 machine에 세팅되어있던 job에서 변경유무 확인
        jop = p_table[0][2]
        if setup_time !=0:
            self.event_list.append((p_table[0][0],"setup", self.r_list[machine], self.time, self.time+setup_time,"setup_change","NONE",step_num,setup_time))
        self.event_list.append((p_table[0][0], jop, self.r_list[machine], self.time, self.time+setup_time+p_table[0][1],"track_in_finish", rule_name, step_num,setup_time))
        self.assign_setting(p_table[0][0], self.r_list[machine],self.time+setup_time+p_table[0][1])
        return p_table[0][1]
        
    def dispatching_rule_SPTSSU(self, machine):
        rule_name= "SPTSSU"
        step_num = self.step_number
        self.step_number+=1
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
                p_table.append([self.j_list[job], self.process_time_table[machine].loc[jop]+setup_time, jop])
        p_table.sort(key = lambda x:x[1], reverse = True)
        setup_time = p_table[0][0].setup_table['j'+str(self.r_list[machine].setup_status)] #컬럼에서 machine에 세팅되어있던 job에서 변경유무 확인
        jop = p_table[0][2]
        if setup_time !=0:
            self.event_list.append((p_table[0][0], "setup", self.r_list[machine], self.time, self.time+setup_time,"setup_change","NONE",step_num,setup_time))
        self.event_list.append((p_table[0][0], jop, self.r_list[machine], self.time, self.time+p_table[0][1],"track_in_finish",rule_name,step_num,setup_time))
        self.assign_setting(p_table[0][0],self.r_list[machine],self.time+p_table[0][1])
        return p_table[0][1] - setup_time
            
    def dispatching_rule_MOR(self,machine):
        rule_name= "MOR"
        step_num = self.step_number
        self.step_number+=1
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
                p_table.append([self.j_list[job], self.process_time_table[machine].loc[jop],jop])
        p_table.sort(key = lambda x:x[0].remain_operation, reverse = True)
        setup_time = p_table[0][0].setup_table['j'+str(self.r_list[machine].setup_status)] #컬럼에서 machine에 세팅되어있던 job에서 변경유무 확인
        jop = p_table[0][2]
        if setup_time !=0:
            self.event_list.append((p_table[0][0],"setup", self.r_list[machine], self.time, self.time+setup_time,"setup_change","NONE",step_num,setup_time))
        self.event_list.append((p_table[0][0],jop, self.r_list[machine], self.time, self.time+setup_time+p_table[0][1],"track_in_finish",rule_name,step_num,setup_time))
        self.assign_setting(p_table[0][0], self.r_list[machine],self.time+setup_time+p_table[0][1])
        return p_table[0][1]
            
    def dispatching_rule_MORSPT(self, machine):
        rule_name= "MORSPT"
        step_num = self.step_number
        self.step_number+=1
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
                p_table.append([self.j_list[job], self.process_time_table[machine].loc[jop],jop])
        p_table.sort(key = lambda x : x[1]/x[0].remain_operation, reverse = False)
        setup_time = p_table[0][0].setup_table['j'+str(self.r_list[machine].setup_status)] #컬럼에서 machine에 세팅되어있던 job에서 변경유무 확인
        jop = p_table[0][2]
        if setup_time !=0:
            self.event_list.append((p_table[0][0], "setup",self.r_list[machine], self.time, self.time+setup_time,"setup_change","NONE",step_num,setup_time))
        self.event_list.append((p_table[0][0], jop,self.r_list[machine], self.time, self.time+setup_time+p_table[0][1],"track_in_finish",rule_name,step_num,setup_time))
        self.assign_setting(p_table[0][0], self.r_list[machine],self.time+setup_time+p_table[0][1])
        return p_table[0][1]
        
            
    def dispatching_rule_LOR(self,machine):
        rule_name= "LOR"
        step_num = self.step_number
        self.step_number+=1
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
                p_table.append([self.j_list[job], self.process_time_table[machine].loc[jop],jop])
        p_table.sort(key = lambda x:x[0].remain_operation, reverse = False)
        setup_time = p_table[0][0].setup_table['j'+str(self.r_list[machine].setup_status)] #컬럼에서 machine에 세팅되어있던 job에서 변경유무 확인
        jop = p_table[0][2]
        if setup_time !=0:
            self.event_list.append((p_table[0][0],"setup", self.r_list[machine], self.time, self.time+setup_time,"setup_change","NONE",step_num,setup_time))
        self.event_list.append((p_table[0][0], jop,self.r_list[machine], self.time, self.time+setup_time+p_table[0][1],"track_in_finish",rule_name,step_num,setup_time))
        self.assign_setting(p_table[0][0], self.r_list[machine],self.time+setup_time+p_table[0][1])
        return p_table[0][1]
        