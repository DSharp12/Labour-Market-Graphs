"""This code will project a weighted Bipartite Graph back onto one of its node set using 
several methods, and then spectrally cluster the graph, and plot the nodes in the space 
spanned by the eigenvectors
"""
import matplotlib.pyplot as plt
import networkx as nx
from networkx import bipartite
import numpy as np
import pandas as pd
from sklearn.cluster import SpectralClustering
from sklearn import metrics

"""This function takes a BiAdjacency Matrix, as a dataframe and returns an Adjacency Matrix 
of a directed or undirected Graph as another dataframe using the projection found in Mealy et al.(2018)
"""

def Mealy_Projection(BiAdj_df, Selfloops=False, Directed=False):
    #BiAdj_df-the dataframe containing the biadjacency matrix
    #Selfloops - boolean for whether adjacency matrix is zero on the diagonal or not, default False    
    #Directed - boolean for whether graph is directed or undirected, default False
    
    
    #initalise row and column indicators
    Vertex1list=BiAdj_df.index.tolist()
    Vertex2list=BiAdj_df.columns.tolist()
    
    
    #1) generate a list of IWA to scarcity
    global Scarcitylist
    Scarcitylist=[]     
    for i in Vertex2list:
        scarcity=1/(BiAdj_df[i].sum())
        Scarcitylist.append(scarcity)
    
      
    #2)#transpose the BiAdjacency Matirx for each of calculation.
    BiAdj_df=BiAdj_df.transpose()
        
    #3) Produce ϕ(i,j) adjacency matrix
    
    
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
                  productSeries=BiAdj_df[i]*BiAdj_df[j]
                  list1=list(productSeries)
                  weighted_product =[a*b for a,b in zip(list1,Scarcitylist)]
                  sum_weighted_product=sum(weighted_product)
                  
                  #Construct a directed or undirected matrix
                  if Directed==True:
                      phi=sum_weighted_product/sum(BiAdj_df[i])
                  else:
                      phi=min(sum_weighted_product/sum(BiAdj_df[i]), sum_weighted_product/sum(BiAdj_df[j]))
              #this list is now ϕ(i)
              phi_list.append(phi)
          else:
              productSeries=BiAdj_df[i]*BiAdj_df[j]
              list1=list(productSeries)
              weighted_product =[a*b for a,b in zip(list1,Scarcitylist)]
              sum_weighted_product=sum(weighted_product)
              
              #Construct a directed or undirected matrix
              if Directed==True:
                  phi=sum_weighted_product/sum(BiAdj_df[i])
              else:
                  phi=min(sum_weighted_product/sum(BiAdj_df[i]), sum_weighted_product/sum(BiAdj_df[j]))
              #this list is now ϕ(i)
              phi_list.append(phi)
              
              
      #turn the list into a series    
      
      phi_series=pd.Series(phi_list)
      
      #once each series is generated, add it to the dataframe 
      df[i]=phi_series.values
          
    Ajd_df=df.drop('init col',axis=1)     
    
    
    return Ajd_df
              
#Undirected_Graph= Mealy_Projection(Adjacencydf)
#Directed_graph=Mealy_Projection(Adjacencydf, Directed=True)





"""This function builds a graph out of any Adjacency Matrix, Spectrally Clusters it
and postions the nodes in the space spanned by the 2nd, 3rd and 4th largest eigenvalues
"""
def SpectralPlot(Adj_df, Directed=False):
    #Adj_df- dataframe of the adjacency matrix of the graph
    
    #get adj matrix as numpy array
    Adj_matrix=Adj_df.values
    
    #cluster the adjacency matrix
    spec_cluster=SpectralClustering(n_clusters=3, n_init=1)0, gamma=1.0, affinity=Adj_matrix)
    
    sc.fit(adj_mat)
    
    #build the graph
    G=nx.from_numpy_matrix(Adj_matrix)
    Eigenvalues = laplacian_spectrum(G, weight='weight')
    





"""This function finds the weighted Katz Centralilty for a Graph with Adjacency Matrix A
"""
def Weighted_Katz(A,alpha,weight_vec):
    #A - Adjacency Matrix of Graph
    #alpha - scaling parameter
    #emplist - a vector to weight the Katz measure by
             
    #Generate the right sized identty matrix
    Id = np.identity(A.shape)
    #note alpha should be smaller than 1/largest eigenvalue of A for traditional katz
    alpha = 0.9
    
    katz_mat =  np.linalg.inv(Id - alpha*A)
        
    katz_weight_cent = katz_mat.dot(weight_vec)
    
    return katz_weight_cent


Undirected_Task_Graph=Mealy_Projection(Task_adj)
Undirected_Skills_Graph=Mealy_Projection(Skills_Adj)

Nodelist=Adjacencydf.index.tolist()
Task_edgelist=nx.to_edgelist(Undirected_Task_Graph,Nodelist)

