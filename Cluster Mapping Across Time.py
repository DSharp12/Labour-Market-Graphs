# -*- coding: utf-8 -*-
"""
Created on Mon Mar 11 17:54:23 2019
This code maps the clusters across time periods of the graph by comparing the clusters with
the highest overlap in occupations
@author: Daniel Sharp
"""

import pandas as pd
import os

filepath =r'C:\Users\Daniel Sharp\OneDrive - Nexus365\MPhil Thesis\Data and Analysis'
os.chdir(filepath)

# import Thesis Module
filepath=r'C:\Users\Daniel Sharp\OneDrive - Nexus365\MPhil Thesis\Data and Analysis\Code\Analysis Modules'
os.chdir(filepath)
import GraphConstruction
import WageAnalysis

# load the cluster identifiers

TaskClusters_2002 = pd.read_excel('2002 Task Weighted Clusters.xlsx')
TaskClusters_2017 = pd.read_excel('TaskweightedClusters.xlsx')

#get OCC_Codes for 2017 Clusters


TaskClusters_2002.rename({'Cluster':'2002 Clusters'}, axis=1,inplace=True)
TaskClusters_2017.rename({'TaskWeighted Cluster':'2017 Clusters'}, axis=1,inplace=True)

# define the clusterlist: 
clusterlist = [i for i in range(10)]

# function to produce a dataframe looking at overlap between clusters
def ClusterOverlap(clusterID1,clusterID2, clustertitle1,clustertitle2):
        
    # align both clusters on one df
    clustermap = GraphConstruction.PandaFormat.df_single_map(clusterID1,
                                                             clusterID2,
                                                             'O*NET-SOC Code',
                                                             clustertitle2)
    
    
    
    
    
    
    