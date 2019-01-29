# -*- coding: utf-8 -*-
"""
Created on Sun Jan  6 23:55:33 2019

@author: Daniel Sharp
This code takes IPUMS data on Unemployment for 463 ACS Code Occupations and Produces 
A Corrolation Matrix between the ACS codes, and run's co-intergration tests on them
"""

#import required modules
import statsmodels.tsa.stattools as ts
import numpy as np
import pandas as pd
import os
from scipy.stats.stats import pearsonr
import networkx as nx
#set filepath
path=r"C:\Users\Daniel Sharp\OneDrive - Nexus365\MPhil Thesis\Data and Analysis"
os.chdir(path)

"""This function takes a dataframe containing time series, and produces a matrix of the 
p-values of the pairwise cointergration tests of each time series
"""
def Coint_Matrix(df):
    #df - the dataframe containing the time series
    
    #initalize corrlation dataframe
    #initalize the column list
    col_list=list(df.columns.values)
    
    
    #initalize the dataframe 
    zerolist=[0]*len(col_list)
    corrdf=pd.DataFrame(data=zerolist,index=col_list, columns=['initcol'])
    
    for i in col_list:
        co_int_list=[]
        for j in col_list:
            if i==j:
                co_int_list.append(0)
            else:    
                x1=Pre_corr_df[i]
                x2=Pre_corr_df[j]
                coin_test=ts.coint(x1,x2)
                co_int_list.append(coin_test[1])
            
        co_int_series=pd.Series(co_int_list)
        
        corrdf[i]=co_int_series.values
    enddf=corrdf.drop('initcol',axis=1)  
    
    return enddf

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

"""This function takes a normal adjacency matrix with non binary elements, and turns it into a
matrix with binary elements according to your theshold
"""
def BinaryAdjMatrix(adj,threshold):
    adj2=adj.copy()
    for column in adj2:
        mask=adj2[column]>threshold
        adj2.loc[mask,column]=1
    
    return adj2

"""This function truncates any matrix (entered as a dataframe) at the nth percentile and returns only the values above that percetile
"""

def PercentileTrunc(df1,n):
    #df - the dataframe containing the matrix
    #n- the percentile above which you want to truncate
    df=df1.copy()
    
    #find the nth percentile
    nthpercentile=np.percentile(df,n, interpolation='midpoint')
    
    #truncate all values below this to 0
    for column in df:
        mask=df[column]<nthpercentile
        df.loc[mask,column]=0
    
    #return the matrix as a dataframe
    return df


"""This function takes an Adjacency Matrix and finds the square n by n matrix of the shortest paths 
between each node and returns it as a dataframe
"""
def ShortPath(adj):
    #adj - adjacency matrix of the graph given as a panda's dataframe
    
    #note that networkx gives you an unlabelled dictionary, but that it is in the same
    #order as the adjacency matrix you where given, so we need to adjust the resulting dataframes indices
    
    #create nx graph object
    adjmat=adj.values
    G=nx.from_numpy_matrix(adjmat)
    #genearte the dictionary of shortest path values
    shortpathdic=dict(nx.shortest_path_length(G))
    #put it into a dataframe
    shortpathdf=pd.DataFrame.from_dict(shortpathdic)
    #change over column labels
    shortpathdf.columns=list(adj.columns)
    shortpathdf.index=list(adj.index)
    
    return shortpathdf

#read in datafiles
Unempdf=pd.read_excel('unemployment_proportion_by_occ_by_month.xlsx')

#clean the dataframe
Pre_corr_df=Unempdf.drop('title',axis=1)
Pre_corr_df.set_index('occ',inplace=True)
Pre_corr_df=Pre_corr_df.transpose()

#generate corrlation matrix
unemp_corr_df=Pre_corr_df.corr()

#generate sequence of truncated corrolation matrices
#if abs(corr(ui,uj)>0.3):
trunc_corr_03=unemp_corr_df
for column in trunc_corr_03:
    mask=abs(trunc_corr_03[column])<0.3
    trunc_corr_03.loc[mask,column]=0

trunc_corr_05=unemp_corr_df
for column in trunc_corr_05:
    mask=abs(trunc_corr_05[column])<0.5
    trunc_corr_05.loc[mask,column]=0
    
"""
THIS CODE IS VERY COMPUTATIONALLY EXPENSIVE, SO USE IT WITH CAUTION
#produce a matrix of co-intergration p-values
#null hypothesis: no-cointergration
col_list=list(Pre_corr_df.columns.values)

#produce the co_intergration matrix
Coint_M=Coint_Matrix(Pre_corr_df)
"""

"""This section of the code looks at the corrolation between the adjacency matrix of our ACS graphs and 
the corrolation matrix. I do not have unemployment time series at the ACS occupational level
"""

#load adjacency matrix of each graph
#graphs made with ACS codes
ACSUnadjdf=pd.read_csv('mariatruncdf.csv')
ACSUnadjdf.drop('Unnamed: 0',axis=1, inplace=True)
ACSUnadjdf2=pd.read_excel('ACS Unweighted Job Space.xlsx')
ACSTaskWadjdf =pd.read_excel('ACS Task Weighted Job Space.xlsx')
ACSSkillWadjdf=pd.read_excel('ACS Skill Weighted Task Job Space.xlsx')

#data cleaning area:
ACSBinary=pd.Dataframe
ACSUnadjdf=ACSUnadjdf.drop('Unnamed: 0',axis=1)


#produce corrolations between each adjacency martrix and the non-truncated corrdf
#quick explination: here i run row level corrolations as the corrolaiton matrix and 
#the adjacency matricies of these graphs are the same

#ACS Unweighted, Skillweighted
ACScorr1=ACSUnadjdf.corrwith(unemp_corr_df)
ACScorr1b=ACSUnadjdf2.corrwith(unemp_corr_df)
ACScorr3=ACSSkillWadjdf.corrwith(unemp_corr_df)   
  

#take simple averages of corr's:
ACSmean1=ACScorr2.mean()
ACSmean2=ACScorr1.mean()
ACSmean3=ACScorr3.mean()

#for task weighted, we need to isolate those columns and rows in the unemployment corrolation
#matrix which align with the Adjacency matrix   
#run the function 
ACSTaskunempcorr=AdjacencySplicer(unemp_corr_df,ACSTaskWadjdf)    
 
#check for corralation
ACScorr2=ACSTaskWadjdf.corrwith(ACSTaskunempcorr) 

#we also produce the truncated reduced unemployment corrolations

task_trunc_corr_03=ACSTaskunempcorr
for column in task_trunc_corr_03:
    mask=abs(task_trunc_corr_03[column])<0.3
    task_trunc_corr_03.loc[mask,column]=0

task_trunc_corr_05=ACSTaskunempcorr
for column in task_trunc_corr_05:
    mask=abs(task_trunc_corr_05[column])<0.5
    task_trunc_corr_05.loc[mask,column]=0


  



#produce corrolations between each truncated adjacnecy matrix and the turncated corrdf
#run each of our adjacency matrices through the truncating mechanism
trunc_ACSUnadjdf=PercentileTrunc(ACSUnadjdf,80)
trunc_ACSTaskWadjdf=PercentileTrunc(ACSTaskWadjdf,80)
trunc_ACSSkillWadjdf=PercentileTrunc(ACSSkillWadjdf,80)

#run the corrolations for corr>0.3
trunc_ACScorr1=trunc_ACSUnadjdf.corrwith(trunc_corr_03)
trunc_ACScorr2=trunc_ACSTaskWadjdf.corrwith(task_trunc_corr_03)
trunc_ACScorr3=trunc_ACSSkillWadjdf.corrwith(trunc_corr_03)

#average the corrolations
trunc_ACSmean1=trunc_ACScorr1.mean()
trunc_ACSmean2=trunc_ACScorr2.mean()
tunrc_ACSmean3=trunc_ACScorr3.mean()

#repeat for corr>0.5    
trunc_ACScorr105=trunc_ACSUnadjdf.corrwith(trunc_corr_05)
trunc_ACScorr205=trunc_ACSTaskWadjdf.corrwith(task_trunc_corr_05)
trunc_ACScorr305=trunc_ACSSkillWadjdf.corrwith(trunc_corr_05)

#average the corrolations
trunc_ACSmean105=trunc_ACScorr105.mean()
trunc_ACSmean205=trunc_ACScorr205.mean()
tunrc_ACSmean305=trunc_ACScorr305.mean()



"""This section of the code looks at the shortest path length across the graph between nodes 
and how that corrolates with the corrolation of their unemployment time series"""

#here we take the truncated graphs and adjust their adjacency matrices to be 0's and 1's

bin_trunc_ACSUnadjdf=BinaryAdjMatrix(trunc_ACSUnadjdf,0.0001)
bin_trunc_ACSTaskWadjdf=BinaryAdjMatrix(trunc_ACSTaskWadjdf,0.0001)
bin_trunc_ACSSkillWadjdf=BinaryAdjMatrix(trunc_ACSSkillWadjdf,0.0001)

#now calculate the shortest path length matrix of each graph
bin_trunc_Unw_Mat=bin_trunc_ACSUnadjdf.values
bin_trunc_Task_Mat=bin_trunc_ACSTaskWadjdf.values
bin_trunc_Skill_Mat=bin_trunc_ACSSkillWadjdf.values

G_Unw=nx.from_numpy_matrix(bin_trunc_Unw_Mat)
G_Task=nx.from_numpy_matrix(bin_trunc_Task_Mat)
G_Skill=nx.from_numpy_matrix(bin_trunc_Skill_Mat)

#create shortest path dataframes
Un_Shortpath=ShortPath(ACSUnadjdf2)
Task_Shortpath=ShortPath(ACSTaskWadjdf)
Skill_Shortpath=ShortPath(ACSSkillWadjdf)

#find the corrolation between the shortest path dataframes and the unemployment corrolation matrix
SP_UnW_corr=Un_Shortpath.corrwith(unemp_corr_df)
SP_Task_corr=Task_Shortpath.corrwith(unemp_corr_df)
SP_Skill_corr=Skill_Shortpath.corrwith(unemp_corr_df)    

#find means    
print(SP_UnW_corr.mean())
print(SP_Task_corr.mean())
print(SP_Skill_corr.mean())













