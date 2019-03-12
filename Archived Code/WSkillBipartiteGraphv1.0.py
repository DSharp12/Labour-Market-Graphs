"""This Code takes O*NET data on Skills and SOC's. It uses the level of skill within each
occupation as weights in a Bipartite Graph, building the graph and returns it and it's
adjacency matrix.
"""

import pandas as pd
import networkx as nx 
import numpy as np
from networkx import bipartite
import matplotlib.pyplot as plt

#define functions to conduct the relevant operations
"""This function deletes duplicate items in a list while retaining its order"""
def list_del(list1):
    #list1 - the list you want to drop duplicates from
    seen = set()
    seen_add = seen.add
    return [x for x in list1 if not (x in seen or seen_add(x))]


"""This function takes a column with multiple types of values for nested groups
and returns single nested groups with multiple columns containnig those values
and the Euclian Norm of these values
"""
def Euclid_norm_sort(df,m1,m2,typecol,valuecol):
    #df - the dataframe in question
    #m1 - column containing first group
    #m2 - column containing second group
    #typecol - the column containing the different type indicators
    #valuecol - the column containing the values of all types
          
    
    #create a dataframe to contain only one value per (m1,m2)nest
    #get your new column values as a list of tuples
    tuple1=list(df[m1])
    tuple2=list(df[m2])
    tuplelist=list(zip(tuple1,tuple2))
    uniquelist=list_del(tuplelist)
    
    #create new dataframe with the unique (m1,m2) map
    df1=pd.DataFrame(uniquelist, columns=[m1,m2])
          
    #create a list of the unique types in typecol
    typelist=list(df[typecol].unique())
    
    #for each type, create new column head and dictionary that has the requisite value
    for i in typelist:
        typedf = df[df[typecol] == i]    
        i_list=list(typedf[valuecol])
        df1[i]=i_list
    
    df1['Euclidian Norm']=np.linalg.norm(df1[typelist].sub(np.array(0)),axis=1)
    
    return df1    
      
"""This function takes a dataframe and builds a weighted bipartite graph"""

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
    global SkillAdj_df
    SkillAdj_df=pd.DataFrame(Adjacency_matrix.toarray())
    SkillAdj_df.index=vertex1
    SkillAdj_df.columns=vertex2
    
    #save dataframe to excel
    ExcelTitle=GraphName+'.xlsx'
    SkillAdj_df.to_excel(ExcelTitle,startrow=0, startcol=0)
        
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
    
#Load the Data
dfskills=pd.read_excel('Skills.xlsx')
#manipulate the Data
dfweight=Euclid_norm_sort(dfskills,'O*NET-SOC Code','Element ID','Scale ID','Data Value')
#build the bipartite graph  
SOC_Skills_bipartite=df_bipartite(dfweight,'O*NET-SOC Code','Element ID','Euclidian Norm','SkillsBipartite')
   