# -*- coding: utf-8 -*-
"""
Created on Tue Feb 12 14:03:49 2019
Cosine Similarity Graphs

@author: Daniel Sharp
"""
#load modules
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import networkx as nx
from networkx import bipartite
import sklearn
import os
from sklearn.cluster import SpectralClustering
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import seaborn as sns
#set filepath
path=r"C:\Users\Daniel Sharp\OneDrive - Nexus365\MPhil Thesis\Data and Analysis"
os.chdir(path)

# import Thesis Module
filepath=r'C:\Users\Daniel Sharp\OneDrive - Nexus365\MPhil Thesis\Data and Analysis\Code\Analysis Modules'
os.chdir(filepath)
import GraphConstruction
import WageAnalysis


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
    #df1-dataframe 1
    #df2-dataframe 2
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

"""This function takes 2 vectors a, b and returns the cosine similarity according 
	to the definition of the dot product
"""
def cos_sim(a, b):
    dot_product = np.dot(a, b)
    norm_a = np.linalg.norm(a)
    norm_b = np.linalg.norm(b)
    cos=dot_product/(norm_a*norm_b)

    return cos

"""This function takes a BiAdjacency Matrix, as a dataframe and returns an Adjacency Matrix 
of a Graph as another dataframe using the cosine projection method.
"""


def Cosine_Projection(BiAdj_df, Selfloops=False, Directed=False):
    #BiAdj_df-the dataframe containing the biadjacency matrix
    #Selfloops - boolean for whether adjacency matrix is zero on the diagonal or not, default False    
    #Directed - boolean for whether graph is directed or undirected, default False
    
    #this prduces the matrix ϕ(i,j)=cos(θ(i,j)) where θ(i,j) is the angle between vectors i and j
    
    #initalise row and column indicators
    Vertex1list=BiAdj_df.index.tolist()
    Vertex2list=BiAdj_df.columns.tolist()
    
     
      
    #1)#transpose the BiAdjacency Matirx for each of calculation (pandas works along columns not rows)
    BiAdj_df=BiAdj_df.transpose()
        
    #2) Produce ϕ(i,j) adjacency matrix
    
    #initalize the dataframe 
    zerolist=[0]*len(Vertex1list)
    df=pd.DataFrame(data=zerolist,index=Vertex1list, columns=['init col'])
      
    for i in Vertex1list:
      #generate a sequence of series ϕ=(ϕ(1),ϕ(2),...ϕ(n)) is a n*n matrix
      #where ϕ(i)=(ϕ(i,1),ϕ(i,2),...ϕ(i,n))' is a 1*n vector
      
      phi_list=[]  
      for j in Vertex1list:
          
          #decide whether or not to include selfloops
          if Selfloops==False:    
              if i==j:
                  phi=0
              else:
                  Vector1=BiAdj_df[i].values
                  Vector2=BiAdj_df[j].values
                  phi=cos_sim(Vector1,Vector2)
              #this list is now ϕ(i)
              phi_list.append(phi)
          else:
              Vector1=BiAdj_df[i].values
              Vector2=BiAdj_df[j].values
              phi=cos_sim(Vector1,Vector2)
              #this list is now ϕ(i)
        
              phi_list.append(phi)
                     
      #turn the list into a series    
      
      phi_series=pd.Series(phi_list)
      
      #once each series is generated, add it to the dataframe 
      df[i]=phi_series.values
    #clear the initalisation column      
    Ajd_df=df.drop('init col',axis=1)     
    
    
    return Ajd_df

"""Take an Adjacency Matrix and Build its Normalized Laplacian """
def Laplacian(Adj):
    #Adj- Adjacency Matrix in a Dataframe format
    
    #Build graph
    G=nx.from_numpy_matrix(Adj.values)
    
    #Take Laplcian
    L=nx.laplacian_matrix(G).asfptype().todense()
    
    #getting node number
    node_num=len(G.nodes())
    
    #degree of each node
    deg=G.degree(weight='weight')
    deg=[deg[i] for i in range(node_num)]
    
    #construct diagonal of sqrt(degree)
    D_Half=np.diag(1./np.sqrt(deg))
    
    #construct normalized Laplacian
    first_dot=np.dot(D_Half,L)
    LapNorm=np.dot(D_Half,first_dot)
    
    return LapNorm


"""This is a function which takes the normalized Laplcian of a Graph,
and returns the orders list of eigenvectors of the Laplacian and runs k-means over them"""

def SpectralLabels(LapNorm,num_clus=10):
    #LapNorm - The normalized Laplacian matrix whose eigenvectors we want to find
    #Number of clusters 
    
    
    #generate the eigenvectors and values
    eigenValues, eigenVectors  = np.linalg.eig (LapNorm)
    
    #generate the list of ordering for eigenvectors
    idx = eigenValues.argsort()  
    eigenValues = eigenValues[idx]
    eigenVectors = eigenVectors[:,idx]
    
    #put them in appropriate data format
    eigenVectors = [eigenVectors[:, i] for i in range(len(eigenVectors))]
    eigenVectors = [np.squeeze(np.asarray(eigenVectors[i])) for i in range(len(eigenVectors))]

    #take the top n eigenvectors and cluster
    #build matrix of the top two
    A = np.column_stack((eigenVectors[1], eigenVectors[2]))
    #add in the rest
    for i in range(num_clus - 2):
        A = np.column_stack((A, eigenVectors[i + 3]))

    #Computing k-means
    kmeans = KMeans(n_clusters=num_clus)
    kmeans.fit(A)
    centroid = kmeans.cluster_centers_
    labels = kmeans.labels_
    
    return labels


"""This function takes the vector of clustering labels, and inserts it into a dataframe of 
node identifiers"""

def SpectralID(labels,Adj,IDdf,m1,c1):
    #labels - the labeled vector of cluster identifiers
    #Adj - the original dataframe
    #LapNorm - the Normalized Laplacian
    #IDdf - the dataframe containing the node identifiers
    #c1 - the column name in IDdf that contains node name
    #m1 - the column name in IDdf that contains the node ID found in Adj
    
   
    #strip the adjacency matrix down to only one column and its index
    Adj1=Adj.iloc[:,0]
    Adj1=Adj1.reset_index()
    
    #add the LabelSeries
    Adj1['Cluster']=pd.Series(labels)
    
    #drop the irrelevant column
    Adj1=Adj1[['index','Cluster']]
    
    #order Adj1 by cluster
    Adj2=Adj1.sort_values('Cluster')
    Adj2.columns=[m1,'Cluster']
    
    #add in the node names from IDdf
    final_df= df_single_map(IDdf,Adj2,m1,c1)
    
    return final_df

"""This function takes the vector of clustering labels, and inserts it into a dataframe of 
node identifiers"""

def SpectralID(labels,Adj,IDdf,m1,c1):
    #labels - the labeled vector of cluster identifiers
    #Adj - the original dataframe
    #LapNorm - the Normalized Laplacian
    #IDdf - the dataframe containing the node identifiers
    #c1 - the column name in IDdf that contains node name
    #m1 - the column name in IDdf that contains the node ID found in Adj
    
   
    #strip the adjacency matrix down to only one column and its index
    Adj1=Adj.iloc[:,0]
    Adj1=Adj1.reset_index()
    
    #add the LabelSeries
    Adj1['Cluster']=pd.Series(labels)
    
    #drop the irrelevant column
    Adj1=Adj1[['index','Cluster']]
    
    #order Adj1 by cluster
    Adj2=Adj1.sort_values('Cluster')
    Adj2.columns=[m1,'Cluster']
    
    #add in the node names from IDdf
    final_df= df_single_map(IDdf,Adj2,m1,c1)
    
    return final_df

#load the dataframes
Un_Bipartite=pd.read_excel('SOC Unweighted Bipartite.xlsx')
Task_Bipartite=pd.read_excel('SOC Task Weighted Bipartite.xlsx')
Occupationdf=pd.read_excel('Occupation Data.xlsx')


#build the cosine similarity graphs
Unweighted_Cos_Sim=Cosine_Projection(Un_Bipartite)
TaskWeighted_Cos_Sim=Cosine_Projection(Task_Bipartite)

#save the dataframes
Unweighted_Cos_Sim.to_excel('SOC Unweighted Cosine Graph.xlsx')
TaskWeighted_Cos_Sim.to_excel('SOC TaskWeighted Cosine Graph.xlsx')

#take the normalized laplacians of the graphs
Un_Lap=Laplacian(Unweighted_Cos_Sim)
Task_Lap=Laplacian(TaskWeighted_Cos_Sim)

#Generate the cluster labels for 10 clusters
Un_labels=SpectralLabels(Un_Lap)
Task_labels=SpectralLabels(Task_Lap)

#construct dataframes of cluster identifiers
Un_Cosine_Clus_df=SpectralID(Un_labels,Unweighted_Cos_Sim,Occupationdf,'O*NET-SOC Code','Title')
Task_Cosine_Clus_df=SpectralID(Task_labels,TaskWeighted_Cos_Sim,Occupationdf,'O*NET-SOC Code','Title')

#save the cluster identifiers to excels
Un_Cosine_Clus_df.to_excel('Un Cosine Clusters (10).xlsx')
Task_Cosine_Clus_df.to_excel('Task Cosine Clusters (10).xlsx')


"""Load Enviroment"""
Cos_clus=pd.read_excel('Un Cosine Clusters (10).xlsx')

Cos_clus=Cos_clus[['O*NET-SOC Code','Cluster']]


"""Run Wage Analysis for Cosine Projection Clusters"""
"""PLEASE NOTE THAT THE CLUSTER NUMBERS DO NOT HAVE THE SAME ASSOCIATION AS THE OTHER
PROJECTION METHODS """
Cos_clus=WageAnalysis.WageFormat.Code_mapper(Cos_clus,'O*NET-SOC Code','OCC_CODE')

#drop the 8 digit codes, and rename the 6 digit codes
Cos_clus.drop(['O*NET-SOC Code'], axis=1,inplace=True)
Cos_clus.drop(['O*NET-SOC Code'], axis=1,inplace=True)
Cos_clus.rename({'OCC_CODE':'O*NET-SOC Code'},axis=1, inplace=True)
Cos_clus.rename({'OCC_CODE':'O*NET-SOC Code'},axis=1, inplace=True)


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

#set renaming dictionaries
rename1={'occ_code':'O*NET-SOC Code', 'tot_emp':'Employment','a_mean':'Av. Wage'}
rename2={'OCC_CODE':'O*NET-SOC Code', 'TOT_EMP':'Employment','A_MEAN':'Av. Wage'}

Renamed1=ColParse(trunc1, rename1)
Renamed2=ColParse(trunc2, rename2)

Renamed1.extend(Renamed2) 
Wagedflist=Renamed1

#generate yearly wageitems
Wage_tseriesdf=dfmerge(Wagedflist,'O*NET-SOC Code',WageYearlist,'Av. Wage')

Cos_clus.rename({'OCC_CODE':'O*NET-SOC Code'},axis=1,inplace=True)
#add on task weighted clusters
Full_Wage_Cluster=df_single_map(Cos_clus,Wage_tseriesdf,'O*NET-SOC Code',
                                'Cluster'
                                )
Full_Wage_Cluster.sort_values(['2004 Av Wage'],ascending=False,inplace=True)
Full_Wage_Cluster.reset_index(inplace=True, drop=True)
Full_Wage_Cluster=Full_Wage_Cluster.drop(Full_Wage_Cluster.index[0:5])
#set data to numeric
for i in WageYearlist:
    Full_Wage_Cluster[i]=pd.to_numeric(Full_Wage_Cluster[i])

applylist=list(Full_Wage_Cluster.columns)
del applylist[20]
del applylist[0]   

#turn wages into log wages
Log_Wage_Cluster=dfapplylog(Full_Wage_Cluster,applylist)

CosCluster_Meandf=ClusterMean(Log_Wage_Cluster, 'Cluster',WageYearlist)

CosCluster_Vardf=ClusterVar(Log_Wage_Cluster, 'Cluster', WageYearlist)

Pandaplt(CosCluster_Meandf,'Year',['Whole Sample','Cluster: 0.0','Cluster: 1.0','Cluster: 4.0','Cluster: 5.0','Cluster: 6.0','Cluster: 7.0','Cluster: 8.0','Cluster: 9.0'])
Pandaplt(CosCluster_Vardf,'Year',['Whole Sample','Cluster: 0.0','Cluster: 1.0','Cluster: 4.0','Cluster: 5.0','Cluster: 6.0','Cluster: 7.0','Cluster: 8.0','Cluster: 9.0'],mark='.')


CosCluster_Rel_Meandf=Normalisedf(CosCluster_Meandf)
CosCluster_Rel_Vardf=Normalisedf(CosCluster_Vardf)
#re-adjust Year
CosCluster_Rel_Meandf['Year']=[i for i in range(1999,2018)]
CosCluster_Rel_Vardf['Year']=[i for i in range(1999,2018)]

Pandaplt(CosCluster_Rel_Meandf,'Year',['Whole Sample','Cluster: 0.0','Cluster: 1.0','Cluster: 4.0','Cluster: 5.0','Cluster: 6.0','Cluster: 7.0','Cluster: 8.0','Cluster: 9.0'])
Pandaplt(CosCluster_Rel_Vardf,'Year',['Whole Sample','Cluster: 0.0','Cluster: 1.0','Cluster: 4.0','Cluster: 5.0','Cluster: 6.0','Cluster: 7.0','Cluster: 8.0','Cluster: 9.0'],mark='.')

#look at wage and varience changes relative to the whole sample

CosCluster_Rel_Meandf['Cluster: 4.0 Diff'] = CosCluster_Rel_Meandf['Cluster: 4.0']-CosCluster_Rel_Meandf['Whole Sample']
CosCluster_Rel_Meandf['Cluster: 0.0 Diff'] = CosCluster_Rel_Meandf['Cluster: 0.0']-CosCluster_Rel_Meandf['Whole Sample']
CosCluster_Rel_Meandf['Cluster: 5.0 Diff'] = CosCluster_Rel_Meandf['Cluster: 5.0']-CosCluster_Rel_Meandf['Whole Sample']
CosCluster_Rel_Meandf['Cluster: 6.0 Diff'] = CosCluster_Rel_Meandf['Cluster: 6.0']-CosCluster_Rel_Meandf['Whole Sample']
CosCluster_Rel_Meandf['Cluster: 7.0 Diff'] = CosCluster_Rel_Meandf['Cluster: 7.0']-CosCluster_Rel_Meandf['Whole Sample']
CosCluster_Rel_Meandf['Cluster: 8.0 Diff'] = CosCluster_Rel_Meandf['Cluster: 8.0']-CosCluster_Rel_Meandf['Whole Sample']
CosCluster_Rel_Meandf['Cluster: 9.0 Diff'] = CosCluster_Rel_Meandf['Cluster: 9.0']-CosCluster_Rel_Meandf['Whole Sample']
#do the same for varience
CosCluster_Rel_Vardf['Cluster: 4.0 Diff'] = CosCluster_Rel_Vardf['Cluster: 4.0']-CosCluster_Rel_Vardf['Whole Sample']
CosCluster_Rel_Vardf['Cluster: 0.0 Diff'] = CosCluster_Rel_Vardf['Cluster: 0.0']-CosCluster_Rel_Vardf['Whole Sample']
CosCluster_Rel_Vardf['Cluster: 5.0 Diff'] = CosCluster_Rel_Vardf['Cluster: 5.0']-CosCluster_Rel_Vardf['Whole Sample']
CosCluster_Rel_Vardf['Cluster: 6.0 Diff'] = CosCluster_Rel_Vardf['Cluster: 6.0']-CosCluster_Rel_Vardf['Whole Sample']
CosCluster_Rel_Vardf['Cluster: 7.0 Diff'] = CosCluster_Rel_Vardf['Cluster: 7.0']-CosCluster_Rel_Vardf['Whole Sample']
CosCluster_Rel_Vardf['Cluster: 8.0 Diff'] = CosCluster_Rel_Vardf['Cluster: 8.0']-CosCluster_Rel_Vardf['Whole Sample']
CosCluster_Rel_Vardf['Cluster: 9.0 Diff'] = CosCluster_Rel_Vardf['Cluster: 9.0']-CosCluster_Rel_Vardf['Whole Sample']

#plot relative to whole sample wages and whole sample varience
Pandaplt(CosCluster_Rel_Meandf,'Year',['Cluster: 0.0 Diff','Cluster: 4.0 Diff','Cluster: 5.0 Diff','Cluster: 6.0 Diff','Cluster: 7.0 Diff','Cluster: 8.0 Diff','Cluster: 9.0 Diff'])
Pandaplt(CosCluster_Rel_Vardf,'Year',['Cluster: 0.0 Diff','Cluster: 4.0 Diff','Cluster: 5.0 Diff','Cluster: 6.0 Diff','Cluster: 7.0 Diff','Cluster: 8.0 Diff','Cluster: 9.0 Diff'],mark='.')


#clusters from unweighted
Cos_Cluster0=Full_Wage_Cluster[Full_Wage_Cluster['Cluster']==0]
Cos_Cluster1=Full_Wage_Cluster[Full_Wage_Cluster['Cluster']==1]
Cos_Cluster2=Full_Wage_Cluster[Full_Wage_Cluster['Cluster']==2]
Cos_Cluster3=Full_Wage_Cluster[Full_Wage_Cluster['Cluster']==3]
Cos_Cluster4=Full_Wage_Cluster[Full_Wage_Cluster['Cluster']==4]
Cos_Cluster5=Full_Wage_Cluster[Full_Wage_Cluster['Cluster']==5]
Cos_Cluster6=Full_Wage_Cluster[Full_Wage_Cluster['Cluster']==6]
Cos_Cluster7=Full_Wage_Cluster[Full_Wage_Cluster['Cluster']==7]
Cos_Cluster8=Full_Wage_Cluster[Full_Wage_Cluster['Cluster']==8]
Cos_Cluster9=Full_Wage_Cluster[Full_Wage_Cluster['Cluster']==9]


#define wage vectors   
Wage_1999vec=Full_Wage_Cluster['1999 Av Wage'].values
Wage_2008vec=Full_Wage_Cluster['2008 Av Wage'].values
Wage_2017vec=Full_Wage_Cluster['2017 Av Wage'].values
#define wage vectors for clusters an d#2 years
#0
Wage_Cos0_2017=Cos_Cluster0['2017 Av Wage'].values
Wage_Task0_2008=Cos_Cluster0['2008 Av Wage'].values
Wage_Task0_1999=Cos_Cluster0['1999 Av Wage'].values
#1
Wage_Task1_2017=Cos_Cluster1['2017 Av Wage'].values
Wage_Task1_2008=Cos_Cluster1['2008 Av Wage'].values
Wage_Task1_1999=Cos_Cluster1['1999 Av Wage'].values
#3
Wage_Task3_2017=Cos_Cluster3['2017 Av Wage'].values
Wage_Task3_2008=Cos_Cluster3['2008 Av Wage'].values
Wage_Task3_1999=Cos_Cluster3['1999 Av Wage'].values
#4
Wage_Task4_2017=Cos_Cluster4['2017 Av Wage'].values
Wage_Task4_2008=Cos_Cluster4['2008 Av Wage'].values
Wage_Task4_1999=Cos_Cluster4['1999 Av Wage'].values
#5
Wage_Task5_2017=Cos_Cluster5['2017 Av Wage'].values
Wage_Task5_2008=Cos_Cluster5['2008 Av Wage'].values
Wage_Task5_1999=Cos_Cluster5['1999 Av Wage'].values
#6
Wage_Task6_2017=Cos_Cluster6['2017 Av Wage'].values
Wage_Task6_2008=Cos_Cluster6['2008 Av Wage'].values
Wage_Task6_1999=Cos_Cluster6['1999 Av Wage'].values
#7
Wage_Task7_2017=Cos_Cluster7['2017 Av Wage'].values
Wage_Task7_2008=Cos_Cluster7['2008 Av Wage'].values
Wage_Task7_1999=Cos_Cluster7['1999 Av Wage'].values
#8
Wage_Task8_2017=Cos_Cluster8['2017 Av Wage'].values
Wage_Task8_2008=Cos_Cluster8['2008 Av Wage'].values
Wage_Task8_1999=Cos_Cluster8['1999 Av Wage'].values
#9
Wage_Task9_2017=Cos_Cluster9['2017 Av Wage'].values
Wage_Task9_2008=Cos_Cluster9['2008 Av Wage'].values
Wage_Task9_1999=Cos_Cluster9['1999 Av Wage'].values


"""Produce Plots of the KDE's of Wages over time"""
sns.distplot(Full_Wage_Cluster['1999 Av Wage'], hist=False)
sns.distplot(Full_Wage_Cluster['2008 Av Wage'],hist=False)
sns.distplot(Full_Wage_Cluster['2017 Av Wage'],hist=False)


#produce a series of per cluster wage distribution plots
#1999 wage distribution of all key clusters
sns.distplot(Cos_Cluster0['1999 Av Wage'],hist=False,kde_kws={'label':'Cluster 0 - 1999'})
sns.distplot(Cos_Cluster1['1999 Av Wage'],hist=False,kde_kws={'label':'Cluster 1 - 1999'})
sns.distplot(Cos_Cluster2['1999 Av Wage'],hist=False,kde_kws={'label':'Cluster 2 - 1999'})
sns.distplot(Cos_Cluster3['1999 Av Wage'],hist=False,kde_kws={'label':'Cluster 3 - 1999'})
sns.distplot(Cos_Cluster4['1999 Av Wage'],hist=False,kde_kws={'label':'Cluster 4 - 1999'})
sns.distplot(Cos_Cluster5['1999 Av Wage'],hist=False,kde_kws={'label':'Cluster 5 - 1999'})
sns.distplot(Cos_Cluster6['1999 Av Wage'],hist=False,kde_kws={'label':'Cluster 6 - 1999'})
sns.distplot(Cos_Cluster7['1999 Av Wage'],hist=False,kde_kws={'label':'Cluster 7 - 1999'})
sns.distplot(Cos_Cluster8['1999 Av Wage'],hist=False,kde_kws={'label':'Cluster 8 - 1999'})
sns.distplot(Cos_Cluster9['1999 Av Wage'],hist=False,kde_kws={'label':'Cluster 9 - 1999'})


#7
sns.distplot(Cos_Cluster7['2017 Av Wage'],hist=False,kde_kws={'label':'Cluster 7 - 2017'})
sns.distplot(Cos_Cluster7['2008 Av Wage'],hist=False,kde_kws={'label':'Cluster 7 - 2008'})
sns.distplot(Cos_Cluster7['1999 Av Wage'],hist=False,kde_kws={'label':'Cluster 7 - 1999'})


#8
sns.distplot(Cos_Cluster8['2017 Av Wage'],hist=False,kde_kws={'label':'Cluster 8 - 2017'})
sns.distplot(Cos_Cluster8['2008 Av Wage'],hist=False,kde_kws={'label':'Cluster 8 - 2008'})
sns.distplot(Cos_Cluster8['1999 Av Wage'],hist=False,kde_kws={'label':'Cluster 8 - 1999'})

#9
sns.distplot(Cos_Cluster9['2017 Av Wage'],hist=False,kde_kws={'label':'Cluster 9 - 2017'})
sns.distplot(Cos_Cluster9['2008 Av Wage'],hist=False,kde_kws={'label':'Cluster 9 - 2008'})
sns.distplot(Cos_Cluster9['1999 Av Wage'],hist=False,kde_kws={'label':'Cluster 9 - 1999'})








