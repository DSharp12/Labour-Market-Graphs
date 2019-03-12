# -*- coding: utf-8 -*-
"""
Created on Sat Jan  5 16:33:07 2019
 Eigen Vector Analysis: 
 In this code I load up the adjacency matricies of three graphs, and analyse the nodes within each
 cluster of the spectrally clustered graphs, I then save the results to an excel file
@author: Daniel Sharp
"""

#import the relvant modules
import pandas as pd
import numpy as np
from numpy import inf
import scipy 
import os
import networkx as nx
import sklearn
from sklearn.cluster import SpectralClustering
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
#set filepath
path=r"C:\Users\Daniel Sharp\Documents\MPhil Thesis\Data and Analysis"
os.chdir(path)

#load up the dataframes
Unw_Graph=pd.read_excel('UnweightedGraph.xlsx')
Taskw_Graph=pd.read_excel('Task_Adj_matrix.xlsx')
Skillw_Graph=pd.read_excel('Skill_Adj_matrix.xlsx')
Occupationdf=pd.read_excel('Occupation Data.xlsx')

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

Un_LapNorm=Laplacian(Unw_Graph)
Task_LapNorm=Laplacian(Taskw_Graph)
Skill_LapNorm=Laplacian(Skillw_Graph)

#spectrally cluster the normalized Laplacians
#first we construct the matrix of ordered eigenvectors 

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

#find the spectral labels for each graph
UnLabels=SpectralLabels(Un_LapNorm)
TaskLabels=SpectralLabels(Task_LapNorm)
SkillLabels=SpectralLabels(Skill_LapNorm)

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
    Adj1.reset_index()
    assert type(Adj1)=='pandas.core.frame.DataFrame'
    
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
  

clusterdf=SpectralID()






"""This function takes a normalized Laplacian of a graph and returns the Spectral Gap Diagram"""

def SpectralGap(LapNorm):
    #NormLap - The Normalized Laplacian
    
    
    #generate the eigenvectors and values
    eigenValues, eigenVectors  = np.linalg.eig (LapNorm)
    
    #generate the list of ordering for eigenvectors
    idx = eigenValues.argsort()  
    eigenValues = eigenValues[idx]
    eigenVectors = eigenVectors[:,idx]
    
    #put them in appropriate data format
    eigenVectors = [eigenVectors[:, i] for i in range(len(eigenVectors))]
    eigenVectors = [np.squeeze(np.asarray(eigenVectors[i])) for i in range(len(eigenVectors))]

    spectral_gap = [eigenValues[i + 1] - eigenValues[i] for i in range(len(eigenValues) - 1)]
    
    #plot the diagram
    plt.plot( [i + 2 for i in range(30)], spectral_gap[:30], "o")
    plt.xlim([0, 30])
    plt.xlabel("cluster")
    plt.ylabel("Spectral gap")
    plt.title("Spectral gap")
    plt.show()








"""Test Enviroment for Spectral Methods"""

test_un=np.array(Un_D_1L)


Un_Spec=SpectralClustering(n_clusters=10, affinity='precomputed',assign_labels="discretize", random_state=0,n_init=100)
Task_Spec=SpectralClustering(n_clusters=10, affinity='precomputed',assign_labels="discretize", random_state=0)


Un_Spec.fit(Un_D_1L)


for i in np.nditer(np.isinf(test_un)):
    if i==True:
        print('inf!')
for i in np.nditer(np.isnan(test_un)):
    if i==True:
        print('inf!')
               

















#producing eigenvectors/values 

Un_eigenValues, Un_eigenVectors  = np.linalg.eig (Un_D_1L)
Task_eigenValues, Task_eigenVectors  = np.linalg.eig (Task_D_1L)
Skill_eigenValues, Skill_eigenVectors  = np.linalg.eig (Skill_D_1L)




#produce spectral_gap graphs (COME BACK TO THIS)
idx = eigenValues.argsort()#[::-1]   
eigenValues = eigenValues[idx]
eigenVectors = eigenVectors[:,idx]
#putting them in the appropriate data format
eigenVectors = [eigenVectors[:, i] for i in range(len(eigenVectors))]
eigenVectors = [np.squeeze(np.asarray(eigenVectors[i])) for i in range(len(eigenVectors))]
#spectral gap
spectral_gap = [eigenValues[i + 1] - eigenValues[i] for i in range(len(eigenValues) - 1)]





#Label the nodes with the SOC Code and ASC Code ID's
#generate the list of nodes
nodelist=Taskw_Graph.index
for i in enumerate(nodelist):
    TaskG.node[i]['Name']=



#Spectrally Cluster Graphs 





