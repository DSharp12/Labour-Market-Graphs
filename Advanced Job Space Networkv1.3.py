"""This code will project a weighted Bipartite Graph back onto one of its node set using 
several methods, and then spectrally cluster the graph, and plot the nodes in the space 
spanned by the eigenvectors
"""
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import networkx as nx
from networkx import bipartite
import numpy as np
import pandas as pd

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
    #produce a list of task scarcity weights sw
    for i in Vertex2list:
        scarcity=1/(BiAdj_df[i].sum())
        Scarcitylist.append(scarcity)
    
      
    #2)#transpose the BiAdjacency Matirx for each of calculation (pandas works along columns not rows)
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
    #clear the initalisation column      
    Ajd_df=df.drop('init col',axis=1)     
    
    
    return Ajd_df
              


"""This function builds a graph out of any Adjacency Matrix, Spectrally Clusters it
and postions the nodes in the space spanned by the 2nd, 3rd and 4th largest eigenvalues
it also saves each nodes position alonge each dimension as a vector of values 1xn where
n is the number of nodes
"""
def SpectralPlot(Adj_df, Figname,Directed=False):
    #Adj_df- dataframe of the adjacency matrix of the graph
    #Directed - Boolean for whether or not the graph is directed
    
    
    
    #get Adjacency Matrix as numpy array
    Adj_matrix=Adj_df.values
    
    #produce NetworkX Graph from Adjacency Matrix
    G=nx.from_numpy_matrix(Adj_matrix)
    
    #Plot two layouts of the graph, one spring weighted, the other a 2D spectral
    nx.draw_spectral(G)
    nx.draw_spring(G)
    
    
    #Generate the 3 dimensional spectral layout of G as a dictionary for each node
    spectral_dict=nx.spectral_layout(G, dim=3, weight='weight', scale=1)
    """add in something here if the graph is directed or not"""
    
    
    #construct x,y,z cordinate lists
    #initalise the lists
    global x
    global y
    global z
    x=[]
    y=[]
    z=[]
    
    #get list of nodes: 
    nodelist=list(spectral_dict.keys())
    #for each node in the list, append x,y,z
    for node in nodelist:
        x.append(spectral_dict.get(node).item(0))
        y.append(spectral_dict.get(node).item(1))
        z.append(spectral_dict.get(node).item(2))
    
    
    #plot each node in 3D space from the dictionary:
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.set_xlim3d(-0.005, 0.005)
    ax.set_ylim3d(-0.005,0.005)
    ax.set_zlim3d(-0.005,0.005)
    ax.scatter(x, y, z, zdir='z', c= 'red')
    Save_title=Figname+'.png'
    fig.savefig(Save_title, dpi=fig.dpi)
    
    #save the output to a file:
    

"""
This function will take a dataframe filled with a Graphs nodes and their attributes,
and produce a set of corrolations of those attributes with a certain number of eigenvectors
"""

def SpectralCorr(df,Adj,c1,Attributelist,Numspec):
    #df - the dataframe containing the nodelist and their attributes
    #Adj - the adjacency matrix of the graph
    #c1 - the column containing the node identifiers in the df and Adj
    #Attributelist - the list of column headers from your dataframe to save as node attributes
    #Numspec - the interger number of dimensions of the spectral clustering
   
    #initalize enviroment: 
     
        
    #get dictionary of attribute vectors from dataframe 
    df.drop_duplicates(subset=c1,keep='first',inplace=True)
    nodelist=list(Adj.index)
        
    dict_att = dict()
    for att in Attributelist:
        list2=list(df[att])
        dict_att[att]=dict(zip(nodelist,list2))
        
    
    #construct cordinate lists in a dictionary for the spectral dimensions
    #initalise the dictionary
    dict_specdim = dict()
    for i in range(Numspec):
        dict_specdim[i] = []
    
     
    #build graph and spectrally cluster:
    
    #get Adjacency Matrix as numpy array
    Adj_matrix=Adj.values
    
    #produce NetworkX Graph from Adjacency Matrix
    G=nx.from_numpy_matrix(Adj_matrix)
    list1=[]
    for i in range(len(nodelist)):
        list1.append(i)
    node_dict=dict(zip(list1,nodelist))
    G=nx.relabel_nodes(G,node_dict)
    #Plot two layouts of the graph, one spring weighted, the other a 2D spectral
    nx.draw_spectral(G)
    nx.draw_spring(G)
    
    
    #Generate the 3 dimensional spectral layout of G as a dictionary for each node
    spectral_dict=nx.spectral_layout(G, dim=Numspec, weight='weight', scale=1)
    
    
    #get list of nodes: 
    
    
    #for each node in the list, append to each of the initalized lists that nodes position
    for i in range(Numspec):
        for node in nodelist:
            dict_specdim[i].append(spectral_dict.get(node).item(i))
        
    #produce the corralations      
    for att in dict_att:
        for specdim in dict_specdim:
            a=list(dict_att[att].values())
            b=list(dict_specdim[specdim])
            #come back and change
            
            #check about length
            print('alength:'+str(len(a)))
            print('blength:' + str(len(b)))
                       
            print('This is the corrolation matrix between '+ att + ' and spectal dimension '+str(specdim))
            print(scipy.stats.pearsonr(a, b))
                    
            #np.corrcoef(a,b))
      
    #print to a txt file    
    #with open('Spectral Corrolations.txt', 'w') as f:
     #   print('Spectral Corrolations.txt', )
    



"""This function finds the weighted Katz Centralilty for each node in a Graph with Adjacency Matrix A
"""
def Weighted_Katz(Adj,weight_vec):
    #A - Adjacency Matrix of Graph
    #alpha - scaling parameter
    #emplist - a vector to weight the Katz measure by
             
    #Generate the right sized identty matrix
    Id = np.identity(Adj.shape)
    #note alpha should be smaller than 1/largest eigenvalue of A for traditional katz
    
    katz_mat =  np.linalg.inv(Id - 0.9*A)
        
    katz_weight_cent = katz_mat.dot(weight_vec)
    
    return katz_weight_cent



Undirected_Skill_Graph = Mealy_Projection(SkillAdj_df)
Directed_Skill_Graph=Mealy_Projection(SkillAdj_df,Directed=True)

#SpectralPlot(Undirected_Skill_Graph)




#Skill_w_Katz=Weighted_Katz(SkillAdj_df,weight_vec=2018_emp)


#Directed_graph=Mealy_Projection(Adjacencydf, Directed=True)




"""
Undirected_Task_Graph=Mealy_Projection(Task_adj)
Undirected_Skills_Graph=Mealy_Projection(Skills_Adj)

Nodelist=Adjacencydf.index.tolist()
Task_edgelist=nx.to_edgelist(Undirected_Task_Graph,Nodelist)"""

