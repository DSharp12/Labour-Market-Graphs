# -*- coding: utf-8 -*-
"""
CAUTION: This is a FAKE dataframe to test the code. IT IS NOT CORRECT.
"""

#testataframe generation
import matplotlib.pyplot as plt
import networkx as nx
from networkx import bipartite
import pandas as pd
import random
from itertools import repeat

#get lists of IWA's, SOC's
#this gives a list of IWA's, DWA's
df1=pd.read_excel('DWA Reference.xlsx')
IWAlist=list(df1['IWA ID'].unique())
print('IWAlist length: '+ str(len(IWAlist)))

#this gives a list of all the SOC's
df2 = pd.read_excel('Task Ratings.xlsx')
SOClist=list(df2['O*NET-SOC Code'].unique())
print('SOClist length: '+ str(len(SOClist)))


#generate dataframe using the first 100 SOC codes, with 10 IWA's per SOC

#create SOC, IWA series:
#initialise list
SOCserieslist=[]
IWAserieslist=[]
Weightserieslist=[]
#generate list of the top 100 SOC codes, repeated 10 times
for i in range(100):
    SOCserieslist.append(SOClist[i])
SOCserieslist=[x for item in SOCserieslist for x in repeat(item,10)]
#give each SOC a random assortment of IWA's, and a random weighting from 0 to 1 
for i in range(1000):
    IWAserieslist.append(random.choice(IWAlist))
    Weightserieslist.append(random.uniform(0,1))

#check list dimensions
print(len(SOCserieslist), len(IWAserieslist), len(Weightserieslist))
#turn lists into dataframe
SOCseries=pd.Series(SOCserieslist)
IWAseries=pd.Series(IWAserieslist)
Wseries=pd.Series(Weightserieslist)
testdf=pd.DataFrame({'O*NET-SOC Code':SOCseries.values, 'IWA ID':IWAseries.values,'Weights':Wseries.values})

#rename dataframe like normal
#testdf.rename(columns={'Task ID':'Task Freq'},inplace=True)


#produce the bipartite graph
"""This function takes a dataframe and builds a weighted bipartite graph
"""
def df_bipartite(df1,v1,v2,w,GraphName):
    #df1-dataframe with partite sets as coloumns, and a column of weights
    #v1-column for first partite set
    #v2-column for second partite set
    #w-column of weights
    #GraphName - name of the graph image and file to be saved
    
    
    #initialise the dataframe
    #dfsort=df1.groupby([v1,v2], as_index=False).unique() 
    """CHECK THIS"""
    vertex1=list(df1[v1].unique())
    vertex2=list(df1[v2].unique())
    #list of each half of the tuples that form the ends of an edge, and its weight
    tuple1=list(df1[v1])
    tuple2=list(df1[v2])
    tuple3=list(df1[w])
    #list of weighted edges
    weighted_edgelist=list(zip(tuple1,tuple2,tuple3))
    
    #build the graph
    G=nx.Graph()
    G.add_nodes_from(vertex1, bipartite=0) 
    G.add_nodes_from(vertex2, bipartite=1)
    #add all edges and weights
    G.add_weighted_edges_from(weighted_edgelist)
    
    #generate the Biadjacency matrix, put it into a dataframe, and save as excel
    Adjacency_matrix=bipartite.biadjacency_matrix(G,vertex1,vertex2)
    global Adjacencydf
    Adjacencydf=pd.DataFrame(Adjacency_matrix.toarray())
    Adjacencydf.index=vertex1
    Adjacencydf.columns=vertex2
    
    #save dataframe to excel
    ExcelTitle=GraphName+'.xlsx'
    Adjacencydf.to_excel(ExcelTitle,startrow=0, startcol=0)
        
    #pickle the graph
    GraphTitle=GraphName+'.gpickle'
    nx.write_gpickle(G, GraphTitle)
    
    #draw the graph
    #create positional dictinoary
    pos={}
    #place respective vertices into place
    pos.update((n, (1, i)) for i, n in enumerate(vertex1))
    pos.update((n, (2, i)) for i, n in enumerate(vertex2))
    nx.draw(G, pos=pos)
    Figname=GraphName+'.png'
    plt.savefig(Figname)
    plt.show()

    return G

test_bipartite=df_bipartite(testdf, 'O*NET-SOC Code','IWA ID','Weights', 'TestGraph')
 