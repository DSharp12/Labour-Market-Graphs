# -*- coding: utf-8 -*-
"""
Created on Sat Jan  5 16:33:07 2019
 Eigen Vector Analysis: 
 In this code I load up the adjacency matricies of three graphs, and analyse the nodes within each
 cluster of the spectrally clustered graphs, I then save the results to an excel file
@author: Daniel Sharp
"""
"""Loading Enviroment
"""

#import the relvant modules
import pandas as pd
import numpy as np
from numpy import inf
import scipy 
import math
import os
import networkx as nx
import sklearn
from sklearn.cluster import SpectralClustering
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
#set filepath
path=r"C:\Users\Daniel Sharp\OneDrive - Nexus365\MPhil Thesis\Data and Analysis"
os.chdir(path)

"""Function Enviroment
"""

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


"""This function builds a graph out of any Adjacency Matrix, Spectrally Clusters it
and postions the nodes in the space spanned by k largest eigenvalues and saves this n*k matrix as 
a dataframe.
"""
def Kdim_SpectralPlot(Adj_df,k, Directed=False):
    #Adj_df- dataframe of the adjacency matrix of the graph
    #Directed - Boolean for whether or not the graph is directed
    #k - the number of largest eigenvectors you want to plot the occupations in 
    
    
    #get Adjacency Matrix as numpy array
    Adj_matrix=Adj_df.values
    
    #produce NetworkX Graph from Adjacency Matrix
    G=nx.from_numpy_matrix(Adj_matrix)
    
       
    #Generate the k dimensional spectral layout of G as a dictionary for each node
    spectral_dict=nx.spectral_layout(G, dim=k, weight='weight', scale=1)
    spectral_plotdf=pd.DataFrame.from_dict(spectral_dict)
    spectral_plotdf.columns=list(Adj_df.index)
    
    spectral_plotdf2=spectral_plotdf.transpose()
    return spectral_plotdf2
"""This function produces a dictionary for each Cluster dataframe that maps each Occ Code (key) to 
a cluster number (value) and inserts those cluster identifiers into a df"""
    
def Clusterdict(IDdf,ID,Clus,Enddf):
    #IDdf - dataframe containing the occupation and cluster ID's
    #ID - Occupation ID column header
    #CLus - CLuster ID column header
    #Enddf - the dataframe you want to append the cluster ID's to
    #generate the dictionary
    ClusDict=dict(zip(IDdf[ID],IDdf[Clus]))
    
    Enddf['Cluster ID']=Enddf.index.map(ClusDict)
    Enddf.sort_values(by=['Cluster ID'], inplace=True)
    
    return Enddf

"""This function takes the mean and the standard deviation of the co-ordinates of occupations
in Spectral space of all occupations and also occuaptions within each cluster
"""
def SpanAnalysis(Spandf,k,ClusterID):
    #Spandf - the dataframe containing the spectral co-ordinates of the occupations and ClusterID
    #k the number of Clusters
    #ClusterID - the column name containing the Cluster identifiers
    
    #loop over the k dimensions
    for i in range(k):
        
        #produce the mean and stdev of all occupations in each eigenvector
        series=pd.Series(Spandf[i])
        smean=series.mean()
        sstd=series.std()
        print('Mean Value in EigenVector ' +str(i) + ': ' + str(smean))
        print('Std in EigenVector ' +str(i) +': ' +str(sstd))
    
    #now calculate in each cluster
    for i in range(k): #the i's are the eigenvectors
        meanlist=[]
        stdlist=[]
        for j in range(k):#the j's are the clusters
            clus_series=Spandf.loc[Spandf[ClusterID]==j,i]
            clusmean=clus_series.mean()
            clus_std=clus_series.std()
            meanlist.append(clusmean)
            stdlist.append(clus_std)
            
        print('Cluster means '+str(i)+': ')
        print(meanlist)
        print('Cluster std'+str(i)+': ')
        print(stdlist)
        
        
"""This function takes the mean and the standard deviation of the log co-ordinates of occupations
in Spectral space of all occupations and also occuaptions within each cluster
"""
def LogSpanAnalysis(Spandf,k,ClusterID):
    #Spandf - the dataframe containing the spectral co-ordinates of the occupations and ClusterID
    #k the number of Clusters
    #ClusterID - the column name containing the Cluster identifiers
    
    #get log co-ordinates
    Spandf.applymap(lambda x: abs(x))
    Spandf.applymap(lambda x: math.log(x))
    #loop over the k dimensions
    for i in range(k):
        
        
        #produce the mean and stdev of all occupations in each eigenvector
        series=pd.Series(Spandf[i])
        smean=series.mean()
        sstd=series.std()
        print('Mean Value in EigenVector ' +str(i) + ': ' + str(smean))
        print('Std in EigenVector ' +str(i) +': ' +str(sstd))
    
    #now calculate in each cluster
    for i in range(k): #the i's are the eigenvectors
        meanlist=[]
        stdlist=[]
        for j in range(k):#the j's are the clusters
            clus_series=Spandf.loc[Spandf[ClusterID]==j,i]
            clusmean=clus_series.mean()
            clus_std=clus_series.std()
            meanlist.append(clusmean)
            stdlist.append(clus_std)
            
        print('Cluster means '+str(i)+': ')
        print(meanlist)
        print('Cluster std'+str(i)+': ')
        print(stdlist)        
        
        
        
"""Running Enviroment
"""
#load up the dataframes
Unw_Graph=pd.read_excel('UnweightedGraph.xlsx')
Taskw_Graph=pd.read_excel('Task_Adj_matrix.xlsx')
Skillw_Graph=pd.read_excel('Skill_Adj_matrix.xlsx')
Occupationdf=pd.read_excel('Occupation Data.xlsx')



#load CLuster Identifiers
Unw_G_ClusterID=pd.read_excel('UnweightedClusters.xlsx')
Task_G_ClusterID=pd.read_excel('TaskweightedClusters.xlsx')
Skill_G_ClusterID=pd.read_excel('SkillWeightedClusterdf.xlsx')

#construct the Laplacians
Un_LapNorm=Laplacian(Unw_Graph)
Task_LapNorm=Laplacian(Taskw_Graph)
Skill_LapNorm=Laplacian(Skillw_Graph)

#find the spectral labels for each graph
UnLabels=SpectralLabels(Un_LapNorm)
TaskLabels=SpectralLabels(Task_LapNorm)
SkillLabels=SpectralLabels(Skill_LapNorm)

#construct three dataframes of cluster identifiers
UnWeightedClusterdf=SpectralID(UnLabels,Unw_Graph,Occupationdf,'O*NET-SOC Code','Title')
TaskWeightedClusterdf=SpectralID(TaskLabels,Taskw_Graph,Occupationdf,'O*NET-SOC Code','Title')
SkillWeightedClusterdf=SpectralID(SkillLabels,Skillw_Graph,Occupationdf,'O*NET-SOC Code','Title')

#Save the files as Excels
UnWeightedClusterdf.to_excel('UnweightedClusters.xlsx')
TaskWeightedClusterdf.to_excel('TaskweightedClusters.xlsx')
SkillWeightedClusterdf.to_excel('SkillWeightedClusterdf.xlsx')



#extending the number of clusters for tasks to 18:
UnExtensionLabels=SpectralLabels(Un_LapNorm,num_clus=18)
UnExtensionClusters=SpectralID(UnExtensionLabels,Unw_Graph,Occupationdf,'O*NET-SOC Code','Title')
UnExtensionClusters.to_excel('UnweightedExtension.xlsx')

TaskExtensionLabels=SpectralLabels(Task_LapNorm,num_clus=18)
TaskExtensionClusters=SpectralID(TaskExtensionLabels,Taskw_Graph,Occupationdf,'O*NET-SOC Code','Title')
TaskExtensionClusters.to_excel('TaskExtension.xlsx')

SkillExtensionLabels=SpectralLabels(Skill_LapNorm,num_clus=18)
SkillExtensionClusters=SpectralID(SkillExtensionLabels,Skillw_Graph,Occupationdf,'O*NET-SOC Code','Title')
SkillExtensionClusters.to_excel('SkillExtension.xlsx')

#generate Spectral Gap Diagrams
SpectralGap(Un_LapNorm)
SpectralGap(Task_LapNorm)
SpectralGap(Skill_LapNorm)


"""This section of the code analyses the variation of the spectrally embedded occupations
"""

#find the spectral plot in 10 dimensions
Unw_SpectralSpanMatrix=Kdim_SpectralPlot(Unw_Graph,k=10)
Task_SpectralSpanMatrix=Kdim_SpectralPlot(Taskw_Graph,k=10)
Skill_SpectralSpanMatrix=Kdim_SpectralPlot(Skillw_Graph,k=10)

#add their cluster identifiers
Un_SpecSpan=Clusterdict(Unw_G_ClusterID,'O*NET-SOC Code','Cluster',Unw_SpectralSpanMatrix)
Task_SpecSpan=Clusterdict(Task_G_ClusterID,'O*NET-SOC Code','Cluster',Task_SpectralSpanMatrix)
SkillSpecSpan=Clusterdict(Skill_G_ClusterID,'O*NET-SOC Code','Cluster',Skill_SpectralSpanMatrix)

#find total and within cluster mean and spread within each eigenvector
SpanAnalysis(Un_SpecSpan,10,'Cluster ID')
SpanAnalysis(Task_SpecSpan,10,'Cluster ID')
SpanAnalysis(SkillSpecSpan,10,'Cluster ID')

#check results in logs
LogSpanAnalysis(Un_SpecSpan,10,'Cluster ID')
LogSpanAnalysis(Task_SpecSpan,10,'Cluster ID')
LogSpanAnalysis(SkillSpecSpan,10,'Cluster ID')










