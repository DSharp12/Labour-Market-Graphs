"""This code takes different BLS and O*NET Datasets, and prepares dataframes of node 
attributes for use in the Graph Analysis
"""

import pandas as pd
import numpy as np
import math
import os


#set filepath
path=r"C:\Users\Daniel Sharp\OneDrive - Nexus365\MPhil Thesis\Data and Analysis"
os.chdir(path)

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

def DiffGenerator(df1, column):
    #df - the dataframe containing the characteristics and identifiers for individual nodes
    #column - the column containing the characteristics 
    
    """THIS CODE NEEDS TO BE CHECKED"""
    df=df1.copy()
    #clean dataframe
    df=df[df[column]!= '*']
      
    
    
    #initalise the new dataframe 
    initlist=[0]*len(df[column])
    Diffdf=pd.DataFrame(data=initlist, index=df.index, columns =['init col'])
    #ensure that all elements of the characteristic are numeric
    charlist=list(df[column])
    global intlist
    intlist=[float(x) for x in charlist]
    
    #populate dataframe with differences
    for i in intlist:
        ilist=[]
        for j in intlist:
            diff=i-j
    
            ilist.append(diff)
            
        iSeries=pd.Series(ilist)
        
        Diffdf[i]=iSeries
    Diffdf.drop('init col',axis=1) 
    
    return Diffdf





#feed in the wage data from BLS
wagedf=pd.read_excel('2017BLSWage.xlsx')
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
att_df['Log Wage']=att_df['A_MEAN'].apply(lambda x:math.log1p(x))
att_df=att_df[att_df['Log Wage']!='*']
att_df.drop_duplicates('O*NET-SOC Code',keep='first',inplace=True)
att_df.reset_index(drop=True,inplace=True)








