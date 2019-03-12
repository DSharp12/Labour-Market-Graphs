# -*- coding: utf-8 -*-
"""
Created on Mon Mar 11 19:01:54 2019

@author: Daniel Sharp
"""

# -*- coding: utf-8 -*-
"""
Created on Mon Mar  4 11:19:28 2019
2008-Cluster Construction
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
WAdf=pd.read_csv('2008 WorkActivity.txt',sep='\t', lineterminator='\r')
# drop 'Level' scores, hold 'Intensity' scores
WAdfIm=WAdf[WAdf['Scale ID']=='IM']

#get the list of occupations as a df:
Occ_df_2008 = pd.DataFrame(WAdf['O*NET-SOC Code'].unique())

Occ_df_2008.rename({0:'O*NET-SOC Code'},axis=1,inplace=True)

# construct bipartite graph from 2008
B_G_2008,B_G_biadjmat,Bipartite_2008=GraphConstruction.PandaFormat.df_bipartite(WAdfIm,'O*NET-SOC Code','Element ID','Data Value','2008SOCTaskipartite')

# Construct the full network using the Mealy projection
Adjacency_2008=GraphConstruction.PandaFormat.Mealy_Projection(Bipartite_2008)

# save as a dataframe
# set filepath
savepath=r'C:\Users\Daniel Sharp\OneDrive - Nexus365\MPhil Thesis\Data and Analysis\Adjacency Matrices\SOC 2008'
os.chdir(savepath)
Adjacency_2008.to_excel('TaskWeighted Mealy Projection Adj SOC 2008.xlsx')

#Run the following Analysis on the graph
#find its spectral gap
GraphConstruction.Spectral.SpectralGap(Adjacency_2008)

#quickload area: USED FOR analysis production!!
Adjacency_2008=pd.read_excel('TaskWeighted Mealy Projection Adj SOC 2008.xlsx')

#get the spectral clusters next to O*NET Occupation Description
soc_desc_2008 = pd.read_csv('2008 Job Descriptions.txt', sep ='\t', lineterminator='\r')  
soc_desc_2008.rename({'O*NET-SOC CODE':'O*NET-SOC Code'}, axis=1,inplace=True)
TaskClusters_2008=GraphConstruction.Spectral.SpectralID(Adjacency_2008,soc_desc_2008,'O*NET-SOC Code',c1='TITLE')
TaskClusters_2008 = WageAnalysis.WageFormat.Code_mapper(TaskClusters_2008,'O*NET-SOC Code','OCC_CODE')
TaskClusters_2008.to_excel('TaskWeighted Clusters 2008 Data.xlsx')
