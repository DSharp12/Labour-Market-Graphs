# -*- coding: utf-8 -*-
"""
Created on Sun Dec 16 22:11:11 2018

@author: Daniel Sharp
THIS IS A TEST ENVIROMENT TO WORK OUT HOW TO GET SPEC CORR TO WORK-THEN FIND THE OTHER BUG IN THE CODE
"""
import math
import networkx as nx
import pandas as pd
import numpy as np
from matplotlib import pylab as plt
import scipy.stats
import sklearn
from sklearn.cluster import KMeans


def SpectralCorr2(df,Adj,c1,Attributelist,Numspec):
    #df - the dataframe containing the nodelist and their attributes
    #Adj - the adjacency matrix of the graph
    #c1 - the column containing the node identifiers in the df and Adj
    #Attributelist - the list of column headers from your dataframe to save as node attributes
    #Numspec - the interger number of dimensions of the spectral clustering
    
    #create new df and Adj dataframe to manipulte to not alter function inputs
    Adj1=Adj
    
    
    #to ensure we have the correct vectors, we first freeze the index and ordering of the adj matrix
    Adjorderlist=Adj1.index.tolist()
    
 
    #get Adjacency Matrix as numpy array
    Adj_matrix=Adj1.values
    
    #produce NetworkX Graph from Adjacency Matrix
    G=nx.from_numpy_matrix(Adj_matrix)
    
    """KEY TO GET RIGHT if networkx does not maintain the order then
    this probably the cause of the issue"""
    #label nodes in G with SOC codes 
    list1=[]
    for i in range(len(Adjorderlist)):
        list1.append(i)
    node_dict=dict(zip(list1,Adjorderlist))
    G=nx.relabel_nodes(G,node_dict)
    """Check the nodes in Graph are labelled properly-lists are slightly different
    try and check why-run the assert and see"""
    #assert np.testing.assert_array_equal(Adjorderlist, list(G.nodes))
  
    #here we write in two methods for spectral clustering
    
    #1)generate top n (Numspec) eigenvectors using nx's spectral clustering function:
    #produces a dictionary of key: value pairs as node:[spectral co-ordinates in n dimensional space]
    spectral_dict=nx.spectral_layout(G, dim=Numspec, weight='weight', scale=1)
    
    
    #now we need to only keep those jobs that also appear in our df attribute data
    #generate a list of df node identifiers
    att_nodelist=list(df[c1])
    #new dictionary with only the keys in spectral_dict that show up in the df node's
    slim_specdict = {node: spectral_dict[node] for node in att_nodelist}
    """add in test to ensure new dict is created properly"""
    
    #put slim_specdict into a dataframe to get columns as eigenvectors
    eigendf=pd.DataFrame.from_dict(slim_specdict)
    eigendf=eigendf.transpose()
    #pull node list identifer out and rename to apply df_single_map function
    eigendf.reset_index(inplace=True)
    eigendf.rename(columns={'index':c1},inplace=True)
    
    #map attributes into the eigendf (requires function df_single_map)
    for att in Attributelist:
        eigendf=df_single_map(df,eigendf,c1,att)
    
    #now we can check for the corrolations between att columns and the spectral columns
    for i in range(Numspec):
          for att in Attributelist:
              print('This is the corrolation matrix between '+ att + ' and spectal dimension '+str(i))
              df[i].corr(df[att])
              
            
    
    
    
    
    
    #2)Generate top n (Numspec) eigenvectors manually
    #Taking laplacian
    L = nx.laplacian_matrix (G).asfptype().todense ()
    
    #Normalizing laplacian by D^-1 L
    nodes = G.nodes ()
    n = len (nodes)
    strength = G.degree(weight='weight')
    strength = [strength[i] for i in range(n)]
    D_1  = np.diag(1./np.sqrt(strength))
    D_1L = np.dot(D_1, L)
    D_1LD_1=np.dot(D_1L,D_1)

    #find the eigenvectors/eigenvalues of normalized Laplacian
    eigenValues, eigenVectors  = np.linalg.eig (D_1L)
    idx = eigenValues.argsort()#[::-1]   
    #sorting eigenvalues by size
    eigenValues = eigenValues[idx]
    eigenVectors = eigenVectors[:,idx]
    #putting them in the appropriate data format
    eigenVectors = [eigenVectors[:, i] for i in range(len(eigenVectors))]
    eigenVectors = [np.squeeze(np.asarray(eigenVectors[i])) for i in range(len(eigenVectors))]
    #spectral gap
    spectral_gap = [eigenValues[i + 1] - eigenValues[i] for i in range(len(eigenValues) - 1)]
    
    

    plt.plot( [i + 2 for i in range(30)], spectral_gap[:30], "o")
    #plt.xlim([0, 30])
    plt.xlabel("cluster")
    plt.ylabel("Spectral gap")
    plt.title("Spectral gap")
    plt.show()
    
    k = Numspec #number of clusters
    A = np.column_stack((eigenVectors[1], eigenVectors[2]))
    for i in range(k - 2):
        #print(i + 3)
        A = np.column_stack((A, eigenVectors[i + 3]))

    
    kmeans = KMeans(n_clusters=k)
    kmeans.fit(A)
    centroid = kmeans.cluster_centers_
    labels = kmeans.labels_











def SpectralCorrcheck(df,Adj,c1,Attributelist,Numspec):
    #df - the dataframe containing the nodelist and their attributes
    #Adj - the adjacency matrix of the graph
    #c1 - the column containing the node identifiers in the df and Adj
    #Attributelist - the list of column headers from your dataframe to save as node attributes
    #Numspec - the interger number of dimensions of the spectral clustering
   
    #create new df and Adj dataframe to manipulte
    Adj1=Adj
    
       
    #initalize enviroment: 
    
        
    #get dictionary of attribute vectors from dataframe 
    df.drop_duplicates(subset=c1,keep='first',inplace=True)
    
    """THIS IS AN ISSUE TO RESOLVE""" 
    global cutlist1
    cutlist1=list(set(Adj1.index)-set(df[c1]))
        
    Adj1.drop(cutlist1,axis=0,inplace=True)
    Adj1.drop(cutlist1,axis=1,inplace=True)
    
    #test1: check the SOC cross-over between the Adj and the df is exactly 1 to 1
    dfcheck=list(df[c1])
    pd.DataFrame(df[c1])
    Adjnodelist=list(Adj.index)
    dfcheck['Adjnodes']=np.asarray(Adjnodelist)
    
    assert np.testing.assert_allclose(Adjnodelist, dfcheck)
    
    assert np.testing.assert_array_equal(Adjnodelist, dfcheck)
    
    nodelist=list(Adj1.index)
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
    Adj_matrix=Adj1.values
    
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
            
            corr= scipy.stats.pearsonr(a, b)    
            print(corr)
            


#clean the dataframe
df1=df1[df1['A_MEAN'] !='*']
df1['Log Wage']=df1['A_MEAN'].apply(lambda x:math.log1p(x))
          
SpectralCorrcheck(df1, Undirected_Skill_Graph,'O*NET-SOC Code',['Log Wage','TOT_EMP'],4)           
            
edudf=pd.read_excel(r'C:\Users\Daniel Sharp\Documents\MPhil Thesis\Data and Analysis\Occupation_Education.xlsx') 
edudf.rename(columns={"Associate's degree":'Associates degree',"Bachelor's degree":'Bachelors degree',"Master's Degree":'Masters degree'},inplace=True)

education_dict={'Less than high school diploma':8,'High school diploma or equivalent':12,'Some college, no degree':13,'Associates degree':14,'Bachelors degree':16,'Masters degree':18,'Doctoral or professional degree':21}                                                          

edudf.drop('Occupation Title',axis=1,inplace=True)
edudf.set_index('OCC_CODE',inplace=True)
edudf['Average Eduaction']=edudf.apply(axis=0)







"""Using Maria's spectral corrolation code to generate the results
"""

def MariaEigenvectors(Adj,df,Numspec,att):
    #Adj - the adjacency matrix of the graph
    #Numspec - the interger number of dimensions of the spectral clustering

    #get attributes
    earnings=df[att]
    
    #Making graph
    G = nx.from_numpy_matrix(Adj)
    #Taking laplacia
    L = nx.laplacian_matrix (G).asfptype().todense ()
    #L = nx.normalized_laplacian_matrix(G).asfptype().todense ()
    #Normalizing laplacian by D^-1 L
    nodes = G.nodes ()
    n = len (nodes)
    strength = G.degree(weight='weight')
    strength = [strength[i] for i in range(n)]
    D_1  = np.diag(1./np.sqrt(strength))
    D_1L = np.dot(D_1, L)
    D_1LD_1=np.dot(D_1L,D_1)
    """Do we not normalize by D_1dotLdotD_1?"""
    
    #Getting P 
    G_Adj = nx.adjacency_matrix(G).asfptype().todense ()
    D_1Adj = np.dot(D_1, Adj)
    
    #computing and ordering eigenvectors
    eigenValues, eigenVectors  = np.linalg.eig (D_1L)
    idx = eigenValues.argsort()#[::-1]   
    #sorting eigenvalues by size
    eigenValues = eigenValues[idx]
    eigenVectors = eigenVectors[:,idx]
    #putting them in the appropriate data format
    eigenVectors = [eigenVectors[:, i] for i in range(len(eigenVectors))]
    eigenVectors = [np.squeeze(np.asarray(eigenVectors[i])) for i in range(len(eigenVectors))]
    #spectral gap
    spectral_gap = [eigenValues[i + 1] - eigenValues[i] for i in range(len(eigenValues) - 1)]
    
    

    plt.plot( [i + 2 for i in range(30)], spectral_gap[:30], "o")
    #plt.xlim([0, 30])
    plt.xlabel("cluster")
    plt.ylabel("Spectral gap")
    plt.title("Spectral gap")
    plt.show()
    
    k = Numspec #number of clusters
    A = np.column_stack((eigenVectors[1], eigenVectors[2]))
    for i in range(k - 2):
        #print(i + 3)
        A = np.column_stack((A, eigenVectors[i + 3]))

    
    kmeans = KMeans(n_clusters=k)
    kmeans.fit(A)
    centroid = kmeans.cluster_centers_
    labels = kmeans.labels_



    
    for i in range(1, 18):
        output=scipy.stats.pearsonr(earnings, eigenVectors[i])[1]
        print(output)
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
               
                
            