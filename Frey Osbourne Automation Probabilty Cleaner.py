# -*- coding: utf-8 -*-
"""
Created on Tue Jan  8 09:52:02 2019
This code takes a Web-scraped Excel file of Frey and Osbournes Automation Probabilties and Blinders
offshoring index and cleans the Data into something more managable for analysis. 
@author: Daniel Sharp
"""

#import the relevant files
import pandas as pd
import os
import re
#set filepath

path=r"C:\Users\Daniel Sharp\OneDrive - Nexus365\MPhil Thesis\Data and Analysis"
os.chdir(path)

#load the dataframe
dirtydf=pd.read_excel('FO AutoProb.xlsx')
dirtydf2=pd.read_excel('Blinder Offshoreability Raw Data.xlsx')
#check data type
type(dirtydf.iloc[0,0])

#create parsing algorithim

def Automationparse(df):
    #df- the dataframe whose data you want to clean
    #columnlist - the list of columns with the dirty
    
    #create meta dataframe
    #zerolist=[0]*len(df.index)
    #finaldf=pd.DataFrame(data=zerolist,index=list(df.index),columns=['init col'])
   
    df1=df['Dirty Data'].str.split(n=3,expand=True)
    df1.drop(0,axis=1,inplace=True)
    df1.dropna(inplace=True)
    
    #sort out the corrupted parsed O*NET SOC Codes
    df2=df1[df1[2]=='0']
    df3=df2[3].str.split(n=1,expand=True)
    df3['Auto Prob']=df2[1]
    
    #drop corrupted rows
    df1=df1[df1[2]!='0']
    df1=df1[df1[2]!='1']
   
    df1=df1.rename({1:'Auto Prob',2:'O*NET-SOC Code',3:'Title'},axis=1)   
    df3=df3.rename({0:'O*NET-SOC Code',1:'Title'},axis=1)
    
    #add back df3 to df1
    df4=pd.concat([df1,df3])

    df4=df4[df4['O*NET-SOC Code']!='Label']
    return df4


def Offshorepare(df):
    #df the offshore dataframe you need to parse
    
    #get the long string as one element
    df1=df['Data Col'].str.split(pat=")",n=2,expand=True)
    df1=df1.rename({0:'ID Col',1:'Data Col',2:''},axis=1)
    df2=df1['Data Col'].str.split(n=4,expand=True)
    df3=df1['ID Col'].str.split(pat="(",n=2,expand=True)
    df3['Offshore Index']=df2[1]
    df3=df3.rename({0:'Occupational Title',1:'Pre O*NET-SOC Code'},axis=1)
    df3.dropna(inplace=True)
    joinlist=list(df3['Pre O*NET-SOC Code'])
    joinedlist=[]
    for i in joinlist:
        joinedlist.append('-'.join([i[0:2],i[2:]]))
        
    df3['O*NET-SOC Code']=pd.Series(joinedlist)
    #adjust the SOC codes to the standard format
    df4=df3[['Offshore Index','O*NET-SOC Code']]
    
    return df4

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

"""A function that maps from one dataframe to another with a single key dictionary
"""
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
    #df2.drop('level_1',axis=1,inplace=True)
    return df2  
#parse the dirty Automation data
Automationdf=Automationparse(dirtydf)

#parse the Offshoring Data
Offshoredf=Offshorepare(dirtydf2)    
Offshoredf.dropna(inplace=True)
Offshoredf.sort_values('Offshore Index',inplace=True)

#save to dataframes
Automationdf.to_excel('SOC Automation Probabilites.xlsx')
Offshoredf.to_excel('Blinder Cleaned Offshoreability.xlsx')


