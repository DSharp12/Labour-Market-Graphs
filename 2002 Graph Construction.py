# -*- coding: utf-8 -*-
"""
Created on Mon Mar  4 11:19:28 2019
2002-Cluster Construction
@author: Daniel Sharp
"""

# import python modules
import os
import numpy as np
import pandas as pd
import networkx as nx 
from networkx import bipartite
import matplotlib.pyplot as plt

# import Thesis Module
filepath=r'C:\Users\Daniel Sharp\OneDrive - Nexus365\MPhil Thesis\Data and Analysis\Code\Analysis Modules'
os.chdir(filepath)
import GraphConstruction
import WageAnalysis
loadpath = r'C:\Users\Daniel Sharp\OneDrive - Nexus365\MPhil Thesis\Data and Analysis'
os.chdir(loadpath)

# load data
WAdf=pd.read_csv('2002 WorkActivity.txt',sep='\t', lineterminator='\r')
# drop 'Level' scores, hold 'Intensity' scores
WAdfIm=WAdf[WAdf['Scale ID']=='IM']

#get the list of occupations as a df:
Occ_df_2002 = pd.DataFrame(WAdf['O*NET-SOC Code'].unique())

Occ_df_2002.rename({0:'O*NET-SOC Code'},axis=1,inplace=True)

# construct bipartite graph from 2002
B_G_2002,B_G_biadjmat,Bipartite_2002=GraphConstruction.PandaFormat.df_bipartite(WAdfIm,'O*NET-SOC Code','Element ID','Data Value','2002SOCTaskipartite')

# Construct the full network using the Mealy projection
Adjacency_2002=GraphConstruction.PandaFormat.Mealy_Projection(Bipartite_2002)

# save as a dataframe
# set filepath
savepath=r'C:\Users\Daniel Sharp\OneDrive - Nexus365\MPhil Thesis\Data and Analysis\Adjacency Matrices\SOC 2002'
os.chdir(savepath)
Adjacency_2002.to_excel('TaskWeighted Mealy Projection Adj SOC 2002.xlsx')

#Run the following Analysis on the graph
#find its spectral gap
GraphConstruction.Spectral.SpectralGap(Adjacency_2002)

#quickload area: USED FOR analysis production!!
Adjacency_2002=pd.read_excel('TaskWeighted Mealy Projection Adj SOC 2002.xlsx')

#get the spectral clusters next to O*NET Occupation Description
soc_desc_2002 = pd.read_csv('2002 Job Descriptions.txt', sep ='\t', lineterminator='\r')  
soc_desc_2002.rename({'O*NET-SOC CODE':'O*NET-SOC Code'}, axis=1,inplace=True)
TaskClusters_2002=GraphConstruction.Spectral.SpectralID(Adjacency_2002,soc_desc_2002,'O*NET-SOC Code',c1='TITLE')
TaskClusters_2002 = WageAnalysis.WageFormat.Code_mapper(TaskClusters_2002,'O*NET-SOC Code','OCC_CODE')



"""Area for Cluster Wage Analysis"""

# load in Wage Data to ensure mapping works well
wagedf_2017=pd.read_excel('2017BLSWage.xlsx')
wagedf_2016=pd.read_excel('2016BLSWage.xlsx')
wagedf_2015=pd.read_excel('2015BLSWage.xlsx')
wagedf_2014=pd.read_excel('2014BLSWage.xlsx')
wagedf_2013=pd.read_excel('2013BLSWage.xls')
wagedf_2012=pd.read_excel('2012BLSWage.xls')
wagedf_2011=pd.read_excel('2011BLSWage.xls')
wagedf_2010=pd.read_excel('2010BLSWage.xls')
wagedf_2009=pd.read_excel('2009BLSWage.xls')
wagedf_2008=pd.read_excel('2008BLSWage.xls')
wagedf_2007=pd.read_excel('2007BLSWage.xls')
wagedf_2006=pd.read_excel('2006BLSWage.xls')
wagedf_2005=pd.read_excel('2005BLSWage.xls')
wagedf_2004=pd.read_excel('2004BLSWage.xls')
wagedf_2003=pd.read_excel('2003BLSWage.xls')
wagedf_2002=pd.read_excel('2002BLSWage.xls')
wagedf_2001=pd.read_excel('2001BLSWage.xls')
wagedf_2000=pd.read_excel('2000BLSWage.xls')
wagedf_1999=pd.read_excel('1999BLSWage.xls')

#produce indicator lists
Yearlist=[str(i) for i in range(1999,2018)]
#generate rename list to generate wage - year indicators
WageYearlist=[str(i) + ' Av Wage' for i in Yearlist]
EmpYearlist=[str(i) + ' Employment' for i in Yearlist]

#rename the dataframe columns for ease of use
#truncate wagedflist
trunc1=[wagedf_1999, wagedf_2000, wagedf_2001, 
        wagedf_2002, wagedf_2003, wagedf_2004, 
        wagedf_2005, wagedf_2006, wagedf_2007, 
        wagedf_2008, wagedf_2009
        ]

trunc2=[wagedf_2010, wagedf_2011, wagedf_2012,
        wagedf_2013, wagedf_2014, wagedf_2015,
        wagedf_2016, wagedf_2017
        ]

# set renaming dictionaries and cleab the wage data
rename1={'occ_code':'O*NET-SOC Code', 'tot_emp':'Employment','a_mean':'Av. Wage'}
rename2={'OCC_CODE':'O*NET-SOC Code', 'TOT_EMP':'Employment','A_MEAN':'Av. Wage'}

Renamed1=WageAnalysis.WageFormat.ColParse(trunc1, rename1)
Renamed2=WageAnalysis.WageFormat.ColParse(trunc2, rename2)
Renamed1.extend(Renamed2) 
Wagedflist=Renamed1

# produce dataframe will all of the years wages
Wage_tseriesdf=WageAnalysis.WageFormat.dfmerge(Wagedflist,'O*NET-SOC Code',WageYearlist,'Av. Wage')

# reorganise the occ_codes  in 2002 Clusters
re_occ_list =list(TaskClusters_2002['OCC_CODE'])
clean_occ_list = [x.strip() for x in re_occ_list]
TaskClusters_2002['OCC_CODE'] = clean_occ_list
Wage_tseriesdf.rename({'O*NET-SOC Code':'OCC_CODE'},axis=1,inplace=True)

# add in the cluster identifiers for full time series
Full_Wage_Cluster_2002 = GraphConstruction.PandaFormat.df_single_map(TaskClusters_2002,
                                                                     Wage_tseriesdf,
                                                                     'OCC_CODE',
                                                                     'Cluster'
                                                                     )
#Clean Wage data
Full_Wage_Cluster_2002.sort_values(['2004 Av Wage'],ascending=False,inplace=True)
Full_Wage_Cluster_2002.reset_index(inplace=True, drop=True)
Full_Wage_Cluster_2002=Full_Wage_Cluster_2002.drop(Full_Wage_Cluster_2002.index[0:5])
#set data to numeric
for i in WageYearlist:
    Full_Wage_Cluster_2002[i]=pd.to_numeric(Full_Wage_Cluster_2002[i])

applylist=list(Full_Wage_Cluster_2002.columns)
del applylist[20]
del applylist[0]   

#set identifier for list of column data to turn to logs: 
Log_Wage_Cluster_2002=WageAnalysis.WageFormat.dfapplylog(Full_Wage_Cluster_2002, applylist)









Adj1=Adjacency_2002.copy()
Adj1 = Adj1.iloc[:,0]
Adj1 = Adj1.reset_index()

Adj1['Cluster']=pd.Series(labels)
Adj1 = Adj1[['index','Cluster']]
Adj1.rename({'index':'O*NET-SOC Code'},axis=1,inplace=True)
#map onto wages (don't compare the cluster identifiers)


Task_clus2002=WageAnalysis.WageFormat.Code_mapper(Adj1,'O*NET-SOC Code','OCC_CODE')


Task_clus2002.drop(['O*NET-SOC Code'], axis=1,inplace=True)
wagedf_2002.rename({'occ_code':'OCC_CODE'},axis=1,inplace=True)










