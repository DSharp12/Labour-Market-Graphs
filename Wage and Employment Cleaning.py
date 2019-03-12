"""This code takes different BLS and O*NET Datasets, and prepares dataframes of node 
attributes for use in the Graph Analysis
"""

import pandas as pd
import numpy as np
import math
import os
from scipy.stats.stats import pearsonr


#set filepath
path=r"C:\Users\Daniel Sharp\OneDrive - Nexus365\MPhil Thesis\Data and Analysis"
os.chdir(path)

"""A function creating a single index dictionary from two columns of a dataframe
"""
def df_single_dict(df1,m1,c1):
    #df1-dataframe 1
    #m1-column containing the dictionary key
    #c1-column containing the dictionary values 
    
    #create the list of keys 
    m1list=list(df1[m1])
    c1list=list(df1[c1])
    df1_dict=dict(zip(m1list,c1list))
    return df1_dict


"""This function produces a mapper from a refined SOC ID code to a less refined SOC ID code
"""
def Code_mapper(df,c1,c2):
    #df1 -the dataframe containing c1 the granular codee
    #c1 - column title for granular code
    #c2 - column title for non granular code
    granularlist=list(df[c1])
    non_granularlist=[]
    for element in granularlist:
        #the 8 didgit SOC is split from the 6 digit by a '.'
        non_granularlist.append(element.split(".")[0])
    df[c2]=non_granularlist
    
    return df

def df_single_map(df1,df2,m1,c1):
    #df1-dataframe 1 exporting column values
    #df2-dataframe 2 importing column values
    #m1-column containing the map, in both dataframes
    #c1-column containing values you want to map
    
    #take the single key 
    df1_dict=df_single_dict(df1,m1,c1)
    #map these values across
    df2[c1]=df2[m1].map(df1_dict)
    #test for poor mapping
    print(df2.isna().sum())
    df2.dropna(inplace=True)
    return df2 

"""This function takes a numerical vector of characteristics and takes the pairwise difference between
the characteristics to generate a matrix
"""

def DiffGenerator(df1, column, idcol):
    #df - the dataframe containing the characteristics and identifiers for individual nodes
    #column - the column containing the characteristics 
    #idcol - the column of item identifiers to table matrix with

    df=df1.copy()
    
    #initalise the new dataframe 
    initlist=[0]*len(df[column])
    Diffdf=pd.DataFrame(data=initlist, index=df.index, columns =['init col'])
    
    #ensure that all elements of the characteristic are numeric
    intlist=list(df[column])
    

   
    
    #populate dataframe with differences
    for i,k in enumerate(intlist):
        i_list=[]
        
        for j,k in enumerate(intlist):
            
            diff=intlist[i]-intlist[j]
    
            i_list.append(diff)
            
        
        
        Diffdf[str(i)]=np.asarray(i_list)
        
    Diffdf.drop('init col',axis=1, inplace=True) 
    
    #set the column and index identifiers 
    id_list=list(df[idcol])
    Diffdf.columns=id_list
    Diffdf.index=id_list
    return Diffdf

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

def AdjacencySplicer2(df1,df2):
    #NOTE: this is a customised versionf of AdjacencySplicer, it takes the two dataframes and finds the
    #intersection of their indicies and cuts both down to that intersection
    #df1 - first dataframe containing adjacency matrix
    #df2 - second dataframe containing adjacency matrix
    
    dflist=list(df1.index)
    adjlist=list(df2.index)
    
    #take the intersection of the two lists
    intersection=list(set(dflist) & set (adjlist))
    
    #cut the columns down in first df
    df1_2=df1[intersection]
    df1_3=df1_2.transpose()
    #cut the rows down
    df1_4=df1_3[intersection]
    
    #cut the columns down in the second df
    df2_2=df2[intersection]
    df2_3=df2_2.transpose()
    #cut the rows down
    df2_4=df2_3[intersection]
    

    return df1_4,df2_4

"""Run Enviroment"""

#feed in the wage data from BLS
wagedf=pd.read_excel('2017BLSWage.xlsx')
#feed in the adjacency matricies



#feed in the skills data from O*NET
skilldf = pd.read_excel('Skills.xlsx', header=0)

att_df=Code_mapper(skilldf,'O*NET-SOC Code','OCC_CODE')
#add column of the 6 didgit SOC total employment
att_df=df_single_map(wagedf,att_df,'OCC_CODE', 'TOT_EMP')
#add column of 6 didgit SOC average wage
att_df=df_single_map(wagedf,att_df,'OCC_CODE', 'A_MEAN')
att_df=df_single_map(wagedf,att_df,'OCC_CODE','A_MEDIAN')
att_df=att_df[att_df['A_MEAN'] !='*']

#add a column with the log of the mean wage
att_df['Log Wage']=att_df.loc[:,'A_MEAN'].apply(lambda x:math.log1p(x))
att_df=att_df[att_df['Log Wage']!='*']
att_df.drop_duplicates('O*NET-SOC Code',keep='first',inplace=True)
att_df.reset_index(drop=True,inplace=True)

#produce a wage differential matrix
#copy over the attribute dataframe
pre_wage=att_df.copy()
pre_wage=pre_wage[['O*NET-SOC Code','Log Wage']]
wage_df=pre_wage.drop_duplicates()

#produce difference matrix
Wagediff=DiffGenerator(wage_df, 'Log Wage','O*NET-SOC Code')

#look at the corrolations between log wage difference and simialrity
wage_diff_vec=Wagediff.values.flatten()

newfilepath=r'C:\Users\Daniel Sharp\OneDrive - Nexus365\MPhil Thesis\Data and Analysis\Adjacency Matrices\SOC'
os.chdir(newfilepath)

#load the SOC adjacency matricies
Skilladj=pd.read_csv('SkillGraph.csv')
Taskadj=pd.read_excel('Task_Adj_matrix.xlsx')
Unadj=pd.read_excel('UnweightedGraph.xlsx')

Skilladj.set_index('Unnamed: 0', inplace=True,drop=True)
#cut adjacency matrices down to size
spli_skill=AdjacencySplicer(Skilladj,Wagediff)
task_adj,task_wagediff=AdjacencySplicer2(Wagediff,Taskadj)
spli_un=AdjacencySplicer(Unadj,Wagediff)

#flatten similarity measures
skill_vec=spli_skill.values.flatten()
task_vec=task_adj.values.flatten()
wage_t_vec=task_wagediff.values.flatten()
Un_vec=spli_un.values.flatten()











