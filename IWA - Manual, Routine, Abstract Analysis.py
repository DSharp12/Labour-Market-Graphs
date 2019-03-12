# -*- coding: utf-8 -*-
"""
Created on Wed Mar  6 12:05:09 2019
This Small Piece of Code produces and Excel with the IWA's and their 
@author: Daniel Sharp
"""

import pandas as pd
import os
import numpy as np
import seaborn as sns
# import Thesis Module
filepath=r'C:\Users\Daniel Sharp\OneDrive - Nexus365\MPhil Thesis\Data and Analysis\Code\Analysis Modules'
os.chdir(filepath)
import GraphConstruction
import WageAnalysis

filepath=r'C:\Users\Daniel Sharp\OneDrive - Nexus365\MPhil Thesis\Data and Analysis'
os.chdir(filepath)

#IWAdf=pd.read_excel('DWA Reference.xlsx')

#IWAdf.drop(['DWA Title','DWA ID'],axis=1,inplace=True)

#IWAdf.drop_duplicates('IWA ID',inplace=True)
#IWAdf.to_excel('IWA - Abstract, Manual, Routine.xlsx')


"""This section of the code uploads the labels IWA's and the original Task-Occupation, 
and examines the proportion of each occupation that does certain types of tasks"""

#load the IWA labels excel file
IWA_label=pd.read_excel('IWA - Abstract, Manual, Routine.xlsx')
#load the occupation data
sociwa = pd.read_excel('SOCIWASkills.xlsx')

#map on to the occupations the scores using dfsinglemap for each column type
map1 = GraphConstruction.PandaFormat.df_single_map(m1='IWA ID',df1=IWA_label,df2=sociwa,c1='Abstract')
map2 = GraphConstruction.PandaFormat.df_single_map(m1='IWA ID',df1=IWA_label,df2=map1,c1='Manual')
map3 = GraphConstruction.PandaFormat.df_single_map(m1='IWA ID',df1=IWA_label,df2=map2,c1='Routine')

# drop unecessary columns
finmap= map3[['O*NET-SOC Code','IWA ID','Abstract','Manual','Routine']]
# get proportion of Abstract, Manual, Routine tasks per occupation
# get the total number of abstract, manural, routine tasks per occupation
numAMR=finmap.groupby('O*NET-SOC Code').sum()
numAMR.reset_index(inplace=True)
numAMR.rename({'index':'O*NET-SOC Code'},axis=1,inplace=True)

#get total number of tasks for each occupation
numuniqtasks = finmap.groupby(by='O*NET-SOC Code', as_index=False).agg({'IWA ID': pd.Series.nunique})

#save these proportions in this dataframe
AMR_propdf=GraphConstruction.PandaFormat.df_single_map(df1=numuniqtasks,
                                                       df2=numAMR,
                                                       m1='O*NET-SOC Code',
                                                       c1='IWA ID'
                                                       )
#normalise each task type by total tasks done:
AMR_propdf['Abstract Proportion'] = AMR_propdf['Abstract']/AMR_propdf['IWA ID']
AMR_propdf['Manual Proportion'] = AMR_propdf['Manual']/AMR_propdf['IWA ID']
AMR_propdf['Routine Proportion'] = AMR_propdf['Routine']/AMR_propdf['IWA ID']

#save as excel
AMR_propdf.to_excel('Abstract, Manual, Routine Task proportions across occupations.xlsx')


"""This section of the code examines the relationship between clusters and proportions of different
types of tasks, and wage percentiles in 2017 and task proportions
"""
# load dataframe of cluster attributes
socclusid=pd.read_excel('TaskweightedClusters.xlsx')
# get the dataframe containing amount of each task type and their clusters.
Cluster_AMR=GraphConstruction.PandaFormat.df_single_map(df1=socclusid,df2=AMR_propdf,
                                                        m1='O*NET-SOC Code',
                                                        c1='TaskWeighted Cluster'
                                                        )
#get average of each over all of the sample
Abstractmean=Cluster_AMR['Abstract Proportion'].mean()
Routinemean=Cluster_AMR['Routine Proportion'].mean()
Manualmean=Cluster_AMR['Manual Proportion'].mean()

# get the average for each column for the clusters
Ab_prop = Cluster_AMR.groupby('TaskWeighted Cluster')['Abstract Proportion'].mean()
Ro_prop = Cluster_AMR.groupby('TaskWeighted Cluster')['Routine Proportion'].mean()
Man_prop = Cluster_AMR.groupby('TaskWeighted Cluster')['Manual Proportion'].mean()
#add the results together
cluster_results = pd.concat([Ab_prop,Ro_prop,Man_prop],axis=1)

#load the wage data for 2017
wagedf_2017=pd.read_excel('2017BLSWage.xlsx')

#clean wage data
# isolate just the wage
cleanwage_2017=wagedf_2017[['OCC_CODE','A_MEAN']]
#set index and set values to numeric
cleanwage_2017.set_index('OCC_CODE',inplace=True)
cleanwage_2017 = cleanwage_2017[cleanwage_2017['A_MEAN']!='*']
cleanwage_2017 = cleanwage_2017.astype(float)
cleanwage_2017.sort_values('A_MEAN',inplace=True)
cleanwage_2017.reset_index(inplace=True)
cleanwage_2017.rename({'index':'OCC_CODE'},axis=1,inplace=True)




#map wages from OCC codes onto Cluster Proportions data
Cluster_AMR=WageAnalysis.WageFormat.Code_mapper(Cluster_AMR,'O*NET-SOC Code','OCC_CODE')
# get the wages and the cluster proportions
AMR_Wage = GraphConstruction.PandaFormat.df_single_map(cleanwage_2017,Cluster_AMR,c1='A_MEAN',m1='OCC_CODE')

AMR_Wage.sort_values('A_MEAN',axis=0,inplace=True)
AMR_Wage.reset_index(inplace=True,drop=True)
#get 
AMR_Wage['2017 Wage Percentile'] = [i/max(list(AMR_Wage.index)) 
                                        for i in list(AMR_Wage.index)
                                        ]

Abs=AMR_Wage['Abstract Proportion']
Rout=AMR_Wage['Routine Proportion']
Man = AMR_Wage['Manual Proportion']

wperc=AMR_Wage['2017 Wage Percentile']

sns.regplot(wperc,Abs,scatter=False,lowess=True, n_boot =5000)
sns.regplot(wperc,Rout,scatter=False,lowess=True, n_boot =5000)
sns.regplot(wperc,Man,scatter=False,lowess=True,n_boot =5000)


Percentile_Rankdf.reset_index(inplace=True, drop=True)

Percentile_Rankdf['1999 Percentile'] = [i/max(list(Percentile_Rankdf.index)) 
                                        for i in list(Percentile_Rankdf.index)
                                        ]



