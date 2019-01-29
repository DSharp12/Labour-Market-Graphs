# -*- coding: utf-8 -*-
"""
Created on Mon Jan 14 20:03:21 2019
This code looks at the performance of my weighted job space against Mealy et al's (2018) version
@author: Daniel Sharp
"""

import statsmodels.tsa.stattools as ts
import numpy as np
import pandas as pd
import os
from scipy.stats.stats import pearsonr
#set filepath
path=r"C:\Users\Daniel Sharp\OneDrive - Nexus365\MPhil Thesis\Data and Analysis"
os.chdir(path)

"""This function takes an adjacency matrix, and, given the same indexing on another dataframe
collapses the other dataframe down such that their indexs and column titles match
"""
def AdjacencySplicer(df,adj):
    #NOTE: df, adj have to have the same indexes, with one an exact subset of the other
    #df - the dataframe which you are splicing
    #adj - the adjacency that have fewer items
    df1=df
    #get the index from the adj as template for changing the df
    adjlist=list(adj.index)
    
    #cut the columns down
    df2=df1[adjlist]
    df3=df2.transpose()
    #cut the rows down
    df4=df3[adjlist]
    
    #check to make sure final dataframe has the same list
    assert list(df4.index)==adjlist
    
    return df4

"""This function changes the diagonal of a square adjacency matrix into your chosen input
and returns the altered adjacency matrix
"""

def DiagChange(adj1,newdiag):
    #adj1 - adjacency matrix whose diagonal you want to change
    #newdiag - the value you want your new diagonal to be
    adj=adj1
    for i in range(len(adj)):
        adj.iloc[i,i]=newdiag
    return adj


#import files
UnweightedAdj=pd.read_excel('mariadf.xlsx')
TaskWadjdf =pd.read_excel('ACS Task Weighted Job Space.xlsx')
SkillWadjdf=pd.read_excel('ACS Skill Weighted Task Job Space.xlsx')
Transistiondf=pd.read_excel('EmpW_tr.xlsm')

#change diagonals in the Task and Skill Adjacency Matrix to not include 
TaskWadjdf=DiagChange(TaskWadjdf,0)
SkillWadjdf=DiagChange(SkillWadjdf,0)


#get index from Skill Matrix
ACS_index=list(SkillWadjdf.index.values)
#add index to Transistion Matrix
Transistiondf.index=ACS_index
Transistiondf.columns=ACS_index

#drop items not in the task adj from the transistions matrix to get a corrolation
Transistiondf2=AdjacencySplicer(Transistiondf,TaskWadjdf)


#turn matricies into an array
trans_array=Transistiondf.values
trans_array2=Transistiondf2.values
Un_array=UnweightedAdj.values
Task_array=TaskWadjdf.values
Skill_array=SkillWadjdf.values

#flatten matricies
trans_vec=trans_array.flatten()
trans_vec2=trans_array2.flatten()
Un_vec=Un_array.flatten()
Task_vec=Task_array.flatten()
Skill_vec=Skill_array.flatten()

#get corrolations
pearsonr(trans_vec,Skill_vec)
pearsonr(trans_vec,Un_vec)
pearsonr(trans_vec2,Task_vec)