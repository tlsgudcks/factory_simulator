# -*- coding: utf-8 -*-
"""
Created on Wed Jan 11 16:19:17 2023

@author: parkh
"""
import pandas as pd
class Job(object):

    # Default Constructor
    def __init__(self, job_id, max_operation, setup_table):
        # Inherited Information
        self.id = job_id #job번호
        self.current_operation_id = 1
        self.status = "WAIT"
        self.max_operation = max_operation
        self.remain_operation = self.max_operation - self.current_operation_id + 1
        # For History
        self.history_list = []
        self.setup_table = setup_table["j"+str(self.id)]
        self.job_flowtime = 0
        self.operation_in_machine = [0 for x in range(max_operation)]
    
    def jop(self):
        jop = ''
        if self.id < 10:
            jop = "j0"+str(self.id)
        else:
            jop = "j"+str(self.id)
        jop = jop+"0"+str(self.current_operation_id)
        return jop
    
    def assign_setting(self, machine):
        machine2 = machine.id
        machine_number = int(machine2[1:])
        self.operation_in_machine[self.current_operation_id - 1] = machine_number
        self.current_operation_id +=1
        self.status = "PROCESSING"
        self.remain_operation -= 1
    
    def complete_setting(self,start_time, end_time, event_type):
        self.status = "WAIT"
        if event_type == "track_in_finish" and self.remain_operation == 0:
            self.job_flowtime += end_time
setup_time_table = pd.read_csv('C:/Users/parkh/FJSP_SETUP_SIM.csv', index_col=(0))
j = Job(1,3,setup_time_table)
print(j)