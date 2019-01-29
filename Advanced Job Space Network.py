"""This code will project a weighted Bipartite Graph back onto one of its node set using 
several methods, and then spectrally cluster the graph, and plot the nodes in the space 
spanned by the eigenvectors
"""

"""This function takes a BiAdjacency Matrix, as a dataframe and returns an Adjacency Matrix 
of a Graph as another dataframe using the projection found in Mealy et al.(2018)
"""
import matplotlib.pyplot as plt
import networkx as nx
from networkx import bipartite
import pandas as pd
import random
from itertools import repeat


def Mealy_Projection(BiAdj_df):
    #BiAdj_df-the dataframe containing the biadjacency matrix
    #vertex1 - the vertex you want to project onto
      
    """CLEAN UP CODE"""      
    #initalise row and column indicators
    Vertex1list=BiAdj_df.index.tolist()
    Vertex2list=BiAdj_df.columns.tolist()
    
    
    #1) generate a list of IWA to scarcity
    global Scarcitylist
    Scarcitylist=[]     
    for i in Vertex2list:
        scarcity=1/sum(BiAdj_df[i])
        Scarcitylist.append(scarcity)
    
    
    #2) generate a dictinoary of SOC's to the sum of their weights
    
    #transpose the BiAdjacency Matirx for each of calculation.
    BiAdj_df=BiAdj_df.transpose()
        
    #3) Produce ϕ(i,j) adjacency matrix
    """CHECK THIS"""
    #initalize the dataframe 
    zerolist=[0]*len(Vertex1list)
    df=pd.DataFrame(data=zerolist,index=Vertex1list, columns=['init col'])
      
    for i in Vertex1list:
      #generate a sequence of series ϕ=(ϕ(1),ϕ(2),...ϕ(n)) is a n*n matrix
      #where ϕ(i)=(ϕ(i,1),ϕ(i,2),...ϕ(i,n))' is a 1*n vector
      
      phi_list=[]  
      for j in Vertex1list:
          productSeries=BiAdj_df[i]*BiAdj_df[j]
          list1=list(productSeries)
          weighted_product =[a*b for a,b in zip(list1,Scarcitylist)]
          sum_weighted_product=sum(weighted_product)
          
          phi=min(sum_weighted_product/sum(BiAdj_df[i]), sum_weighted_product/sum(BiAdj_df[j]))
          #this list is now ϕ(i)
          phi_list.append(phi)
               
      #turn the list into a series    
      phi_series=pd.Series(phi_list)
      
      #once each series is generated, add it to the dataframe 
      df[i]=phi_series.values
          
    Ajd_df=df.drop('init col',axis=1)     
    
    
    return Ajd_df
              
test_adjdf= Mealy_Projection(Adjacencydf)    
        
    
    
    







"""
#5) Pickle the whole file or shelve it #this does
filename= 'WBipartiteGraph'
my_shelf = shelve.open(filename,'n')

#Note to open the data you can use:
my_shelf = shelve.open(filename)"""