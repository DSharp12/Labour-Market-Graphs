"""This code looks at different relationships between tasks, jobs and wages"""
#import the modules
import pandas as pd
import os
import math
#set filepath
path=r"C:\Users\Daniel Sharp\Documents\MPhil Thesis\Data and Analysis"
os.chdir(path)

"""The functions needed for analysis:
"""

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

def df_single_dict(df1,m1,c1):
    #df1-dataframe 1
    #m1-column containing the dictionary key
    #c1-column containing the dictionary values 
    
    #create the list of keys 
    m1list=list(df1[m1])
    c1list=list(df1[c1])
    df1_dict=dict(zip(m1list,c1list))
    return df1_dict

#load the files
SOCIWA=pd.read_excel('SOCIWASkills.xlsx',header=0)
wagedf=pd.read_excel('2017BLSWage.xlsx',header=0)




#feed in the skills data from O*NET
att_df=Code_mapper(SOCIWA,'O*NET-SOC Code','OCC_CODE')
#add column of the 6 didgit SOC total employment
att_df=df_single_map(wagedf,att_df,'OCC_CODE', 'TOT_EMP')
#add column of 6 didgit SOC average wage
att_df=df_single_map(wagedf,att_df,'OCC_CODE', 'A_MEAN')
att_df=df_single_map(wagedf,att_df,'OCC_CODE','A_MEDIAN')
att_df=att_df[att_df['A_MEAN'] !='*']
#add a column with the log of the mean wage
att_df=att_df[att_df['A_MEDIAN']!='#']
analysis_df=att_df
analysis_df=analysis_df[['OCC_CODE','O*NET-SOC Code','IWA ID','A_MEAN','A_MEDIAN','Average Data Value']]
analysis_df=analysis_df.sort_values('IWA ID')


#ensure that your series are the right datatype
analysis_df['A_MEAN']=analysis_df['A_MEAN'].astype(int)
analysis_df['A_MEDIAN']=analysis_df['A_MEDIAN'].astype(int)

#get IWA mean wage
IWA_Mean_Wagelist=list(analysis_df.groupby(['IWA ID'])['A_MEAN'].mean())
IWA_Median_Wagelist=list(analysis_df.groupby(['IWA ID'])['A_MEAN'].mean())
IWAlist=list(analysis_df['IWA ID'].unique())
IWA_Wagedict=dict(zip(IWAlist,IWA_Mean_Wagelist))
#put it back into the dataframe
analysis_df['IWA Wage']=analysis_df['IWA ID'].map(IWA_Wagedict)

#now find the average wage of the occupation as the average of their corresponding IWA Wages
analysis_df=analysis_df.sort_values('OCC_CODE')
OCCMeanWage=list(analysis_df.groupby(['OCC_CODE'])['IWA Wage'].mean())
OCClist=list(analysis_df['OCC_CODE'].unique())
OCC_wagedict=dict(zip(OCClist,OCCMeanWage))
analysis_df['OCC Wage']=analysis_df['OCC_CODE'].map(OCC_wagedict)
analysis_df['diff'] = analysis_df['A_MEAN'] - analysis_df['OCC Wage']

#now we find the mean weighted wage, by normalising the task intensty and then 
#multiplying it across to each IWA wage within each occupation
analysis_df['Norm Task Intensity']=analysis_df['Average Data Value'].map(lambda x:x/5)
analysis_df['Weighted IWA Wage']=analysis_df['IWA Wage']*analysis_df['Norm Task Intensity']
OCCWeightedMeanWage=list(analysis_df.groupby(['OCC_CODE'])['Weighted IWA Wage'].mean())
OCC_wegihtedwagedict=dict(zip(OCClist,OCCWeightedMeanWage))
analysis_df['OCC Weigthed Wage']=analysis_df['OCC_CODE'].map(OCC_wegihtedwagedict)
analysis_df['diff2'] = analysis_df['A_MEAN'] - analysis_df['OCC Weigthed Wage']

#look at histogrames of wages and IWA wages
analysis_df['Log Wage']=analysis_df['A_MEAN'].apply(lambda x:math.log1p(x))
IWA_wagedf=analysis_df[['IWA ID','Log Wage','A_MEAN','IWA Wage']]
IWA_wagedf['Log IWA Wage']=IWA_wagedf['IWA Wage'].apply(lambda x:math.log1p(x))

#plots
IWA_wagedf.hist('Log IWA Wage')
IWA_wagedf.hist('IWA Wage')
analysis_df.hist('A_MEAN')
analysis_df.hist('Log Wage')


