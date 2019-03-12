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
#set filepath
path=r"C:\Users\Daniel Sharp\Documents\MPhil Thesis\Data and Analysis"
os.chdir(path)

#load up the dataframes
Unw_Graph=pd.read_excel('UnweightedGraph.xlsx')
Taskw_Graph=pd.read_excel('Task_Adj_matrix.xlsx')
Skillw_Graph=pd.read_excel('Skill_Adj_matrix.xlsx')

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
Skill_LapNorm=Laplacian(Skill_LapNorm)



#Build three graphs using Networkx Package
UnG=nx.from_numpy_matrix(Unw_Graph.values)
TaskG=nx.from_numpy_matrix(Taskw_Graph.values)
SkillG=nx.nx.from_numpy_matrix(Skillw_Graph.values)

#Taking Laplacians of Graphs
UnL = nx.laplacian_matrix (UnG).asfptype().todense ()
TaskL =nx.laplacian_matrix (TaskG).asfptype().todense ()
SkillL=nx.laplacian_matrix (SkillG).asfptype().todense ()

#Normalising the Laplacians
#getting lists of nodes
UnG_nodes = UnG.nodes ()
TaskG_nodes=TaskG.nodes()   
SkillG_nodes=SkillG.nodes()

#nodelengths
UnG_n = len (UnG_nodes)
TaskG_n= len(TaskG_nodes)
SkillG_n=len(SkillG_nodes)

#degrees
UnG_deg=UnG.degree(weight='weight')
Task_deg=TaskG.degree(weight='weight')
Skill_deg=SkillG.degree(weight='weight')

#construct a list for each node
UnG_deg=[UnG_deg[i] for i in range(UnG_n)]
Task_deg=[Task_deg[i] for i in range(TaskG_n)]
Skill_deg=[Skill_deg[i] for i in range(SkillG_n)]

#constuct the diagonal matrix
Un_D_1  = np.diag(1./np.sqrt(UnG_deg))
Task_D_1  = np.diag(1./np.sqrt(Task_deg))
Skill_D_1  = np.diag(1./np.sqrt(Skill_deg))

#csontruct normalized laplacians
Un_D_1L = np.dot(Un_D_1,UnL)
Task_D_1L=np.dot(Task_D_1,TaskL)
Skill_D_1L=np.dot(Skill_D_1,SkillL)

#spectrally cluster the nodes 
#first we construct the matrix of ordered eigenvectors 

"""This is a function which takes the normalized Laplcian of a Graph,
and returns the orders list of eigenvectors of the Laplacian"""

def SepectralCluster(LapNorm,num_clus=10):
    #LapNorm - The normalized Laplacian matrix whose eigenvectors we want to find
    #Number of clusters 
    
    
    #generate the eigenvectors and values
    eigenValues, eigenVectors  = np.linalg.eig (LapNorm)
    
    #generate the list of ordering for eigenvectors
    idx = eigenValues.argsort()  
    eigenValues = eigenValues[idx]
    eigenVectors = eigenVectors[:,idx]
    

    


#putting them in the appropriate data format
#first put them into a list
Un_eigenVectors = [Un_eigenVectors[:, i] for i in range(len(Un_eigenVectors))]
Un_eigenVectors = [np.squeeze(np.asarray(Un_eigenVectors[i])) for i in range(len(Un_eigenVectors))]


















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





