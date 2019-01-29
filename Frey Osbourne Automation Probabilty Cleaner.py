# -*- coding: utf-8 -*-
"""
Created on Tue Jan  8 09:52:02 2019
This code takes a Web-scraped Excel file of Frey and Osbournes Automation Probabilties and cleans
the Data into something more managable. 
@author: Daniel Sharp
"""

#import the relevant files
import pandas as pd
import os

#set filepath
path=r"C:\Users\Daniel Sharp\Documents\MPhil Thesis\Data and Analysis"
os.chdir(path)

#load the dataframe
dirtydf=pd.read_excel('FO AutoProb.xlsx')

#check data type
type(dirtydf.iloc[0,0])

#create parsing algorithim

def dirtyparse(df,columnlist):
    #df- the dataframe whose data you want to clean
    #columnlist - the list of columns with the dirty
    
    #create meta dataframe
    zerolist=[0]*len(df.index)
    finaldf=pd.DataFrame(data=zerolist,index=list(df.index),columns=['init col'])
    
    for i in columnlist:
        df1=df[i].str.split(n=3,expand=True)
        df1.drop(0,axis=1,inplace=True)
        finaldf.concat(df1,axis=1,sort=False)
    
    return finaldf
        
#parse the dirty data
df1=dirtyparse(dirtydf)

df1.columns=['Automation Probability','O*NET-SOC Code','Title']        

df1.to_excel('INSERT CLEANED TITLE HERE')        
        
        
        
    
    