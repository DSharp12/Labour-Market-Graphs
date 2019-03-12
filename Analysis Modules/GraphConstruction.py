# -*- coding: utf-8 -*-
"""
Created on Mon Mar  4 11:19:21 2019
O*NET Data Cleaning Module
@author: Daniel Sharp
"""

#import relevant files

import numpy as np
import pandas as pd
import networkx as nx 
from networkx import bipartite
import matplotlib.pyplot as plt
import scipy.stats
from sklearn.cluster import KMeans
import math

class PandaFormat(object):
    
    def __init__(self,name):
        self.name = name
        
    
    
    # This function generates a sequence of lists of unique items 
    # from columns in a dataframe
    @classmethod
    def unique_lists(self,df,collist):
        # df - dataframe you are generating lists from
        # collist - list of column names whose unique values you want to generate
        
        # run through each column title in a for loop
        for i in collist:
            nospace = i.replace(" ","")
            # create the name of the list
            listname = nospace+'_list'
            # create the list
            list1 = list(df[i].unique())
            # change the variable to the listname
            exec(listname + "=  list(df[i].unique())")
            
            return list1
        


     # A function that generates a multi_index dictionary from a dataframe
    @classmethod
    def df_multi_dict(self,df1,m1,m2,c1):
        # df1-dataframe 1
        # m1-column containing the first dictionary key
        # m2-column containing the second dictionary key
        # c1-column containing the dictionary values 
       
        # sort the dataframes to generate ordered lists
        df1 = df1.sort_values([m1,m2])
        # generate a list of the first index
        m1list = list(df1[m1]) 
        # generate a list of the second index
        m2list = list(df1[m2])
        # ties indices together in a list of tuples 
        m1m2list = list(zip(m1list,m2list)) 
        # generate the list of values for tuple keys
        c1list = list(df1[c1])
        # create dictionary with tuple as key and column as values 
        df1_dict = dict(zip(m1m2list,c1list)) 
        return df1_dict

    # A function that maps from one dataframe to another, conditional on multiple coloumn values
    @classmethod
    def df_multi_map(self,df1,df2,m1,m2,c1):
        # df1-dataframe 1 - exporting column values
        # df2-dataframe 2 - importing column values
        # m1-first column from map 
        # m2-second (nested) column from map
        # c1-column containing the dictionary values 
       
        # sort values to get ordering
        df2 = df2.sort_values([m1,m2])
        # generate the desired mapping from the first dataframe with a tuple-key dictionary
        df1_dict = PandaFormat.df_multi_dict(df1,m1,m2,c1)  
        # generate the tuple key in the second dataframe
        m1list = list(df2[m1]) 
        m2list = list(df2[m2])
        df2['tuplelist'] = list(zip(m1list,m2list))
        # generate the empty column to fill with called values
        df2[c1]="" 
        # for each tupple in the second dataframe, call the value from the dictionary   
        df2[c1] = df2['tuplelist'].map(df1_dict)    
        df2.drop('tuplelist',axis=1, inplace=True)
        # test for poor mapping
        print(df2.isna().sum())
        df2.dropna(inplace=True)
        
        return df2    
    
    # A function creating a single index dictionary from two columns of a dataframe
    @classmethod
    def df_single_dict(self, df1,m1,c1):
        # df1-dataframe 1
        # m1-column containing the dictionary key
        # c1-column containing the dictionary values 
        
        # create the list of keys 
        m1list = list(df1[m1])
        c1list = list(df1[c1])
        df1_dict = dict(zip(m1list,c1list))
        
        return df1_dict

    # A function that maps from one dataframe to another with a single key dictionary
    @classmethod
    def df_single_map(self, df1,df2,m1,c1):
        # df1-dataframe 1 exporting column values
        # df2-dataframe 2 importing column values
        # m1-column containing the map, in both dataframes
        # c1-column containing values you want to map
        
        # take the single key 
        df1_dict = PandaFormat.df_single_dict(df1,m1,c1)
        # map these values across
        df2[c1] = df2[m1].map(df1_dict)
        # test for poor mapping
        print(df2.isna().sum())
        df2.dropna(inplace=True)
        
        return df2 
    
    # This function conducts a nested sort from a list on a GroupBy object
    @classmethod
    def sort_grp(self, df, list1):
        # we take the dataframe, as sort along the columns
        df = df.sort_values(list1, ascending=[False,False,False])
        
        return df

    
    # This function takes the average value of multiple nested elements in a dataframe
    # and adds them to the dataframe
    @classmethod
    def average_nest(self, df,m1,m2,avglist):
        # df-dataframe
        # m1-fist column to group by
        # m2-second column to group by
        # avglist-list of columns you wish to average
        # avgnames-list of names of averge values column
        
        #initalise the new average columns for the dataframe
        df1 = df #the function does not alter its original input dataframe
        for i in avglist:
            # find the mean of each list element 1 in each nested group
            dfmean = df1.groupby([m1,m2])[i].mean()
            # create a multi-index dictionary with the multi-index as key, average as value        
            list1 = dfmean.index.tolist()
            list2 = dfmean.tolist()
            avg_dict = dict(zip(list1,list2))
            # read the multi-index dictionary into the dataframe
            
            # generate the tuple list to map the average values
            m1list = list(df1[m1]) 
            m2list = list(df1[m2])
            df1['tuplelist'] = pd.Series(list(zip(m1list,m2list)))
            
            
            df1.dropna(axis=0,subset = ['tuplelist'],inplace=True)
            
            
            # pull average value column name from avgnames
            columnname = 'Average '+i 
            df1[columnname] = ""
            # map average values from dictionary to dataframe   
            df1[columnname] = df1['tuplelist'].map(avg_dict)
        # clean up the dataframe    
        df1.drop('tuplelist',axis=1, inplace=True)
        
        return df1

    # This function normalizes nested values in a dataframe
    @classmethod
    def normal_nest_twocol(self,df,m1,m2,nestlist):
        #df-dataframe
        #m1-fist column to group by
        #m2-second column to group by
        #avglist-list of columns you wish to normalise
    
        for i in nestlist:
            
            #create a Groupby series object with the normalizing value
            nestsum=df.groupby([m1,m2])[i].sum()
            
            #create a multi-index dictionary with the multi-index as key, normalized value as value        
            #create a list of the multi-index
            list1=nestsum.index.tolist()
           
            #crate a list of normalizing values and their reciprocal for each (m1,m2) tuple
            sumlist=nestsum.tolist()             
            normaliser=list(np.reciprocal(sumlist))            
            #create dictionary relating each (m1,m2) to a value, and its normalizer    
            normalizer_dict=dict(zip(list1,normaliser))
             
            #read the multi-index dictionary into the dataframe
            
            #generate the tuple list to map the normalizer value
            m1list=list(df[m1]) 
            m2list=list(df[m2])
            df['tuplelist']= pd.Series(list(zip(m1list,m2list)))
            #generate the column containing the normalizer for each (m1,m2) pair
            df['normalizer']=df['tuplelist'].map(normalizer_dict)          
                         
            
            #pull average value column name from nestlist
            columnname='Normalized '+i 
            df[columnname]=""
            
            #generate normalizing column by multiplying the two series together   
            df[columnname]=df['normalizer']*df[i]
            
                    
        #clean up the dataframe    
        df.drop('tuplelist',axis=1, inplace=True)    
        df.drop('normalizer',axis=1,inplace=True)
        
        return df
     

    # drop any repetitions of IWA's within SOC's
    # This function drops duplicates of c1 within a GroupBy Object
    @classmethod
    def within_drop(self,df,c1):
        df=df.drop_duplicates(subset=c1,keep='first')
        return df 


     # This function takes column elements that are lists, and extends the dataframe to include
     # them in their own column
    @classmethod
    def list_extend(self,df,m1,m2,c1):
        # df - dataframe 
        # m1 - first group
        # m2 - second group
        # c1 - column containing lists
        
        # get the nested index to return m1,m2         
        x = df.index[0]
        # get m1,m2 to repeat in the right lengths 
        length = len(df.at[x,c1])
        # set global variables to check intermediary steps in the functions process
        global m1list
        # produce three lists, all of equal length, with each of m1,m2 repeated the size of the df
        m1list = [df.at[x,m1]]*length
        m2list = [df.at[x,m2]]*length
        
        c1list = list(df.at[x,c1])
        tuplelist = zip(m1list,m2list,c1list)
        # turn the zipped lists in  a list of lists
        list_of_tup = [list(elem) for elem in tuplelist]
        # then apply these lists to the three columns in df2
        df2 = pd.DataFrame(list_of_tup, columns=[m1,m2,c1])
        # add those three columns to the bottom of the df, to be stacked in 'groupby.apply'
        df = df.append(df2)
        df = df.iloc[1:]
    
        return df

     # This function takes the BiAdjacency binary matrix and returns a dataframe 
     # with a nested list, for each partite, the elements of the other partite associated with it
    @classmethod
    def Biadj_todf(self,Adj, columnlist):
        # Adj - adjacency matrix
        # columnlist - list of column headers (each partite)
    
        # generate lists of nodes in each vertex
        vertex1list = list(Adj)
        # initalise dataframe
        df = pd.DataFrame(columns=columnlist)   
        
        # generate a dictionary, for each column value, of a dictionary of skills and 
        # binary value associated with it, the drop all the key,value pairs where value is 0
        meta_dict = Adj.to_dict()
        
        for element in vertex1list:
            # drop items where value is 0
            meta_dict[element] = { k:v for k, v in meta_dict[element].items() if v==1 }
            # get list of the nodes in vertex two associated with node: element
            v2sublist = list(meta_dict[element].keys())
            # add multiples of vertex1 to the dataframe 
            elementlist = [element]*len(v2sublist)
            tuplelist = zip(elementlist,v2sublist)
            list_of_tup = [list(elem) for elem in tuplelist]
            df2 = pd.DataFrame(list_of_tup, columns=columnlist)
            df = df.append(df2)
            
        return df


    # construct the weights off the rank or datavalue of each IWA in each Occupation



    # check to see any repetitions of IWA's in each SOC
    # numiwa=finaldf.groupby(by=['O*NET-SOC Code', 'IWA ID'], as_index=False).count()[['O*NET-SOC Code','DWA ID', 'IWA ID']]
    # numiwa.rename(columns={'DWA ID':'IWA Freq'},inplace=True)
 
 
class GraphBuilder(object):

    def __init__(self,name):
        self.name=name
   
    #4) Build the weighted bipartite graph from the dataframe
    # This function takes a dataframe and builds a weighted bipartite graph
    @classmethod
    def df_bipartite(self,df1,v1,v2,w,GraphName):
        # df1-dataframe with partite sets as columns, and a column of weights
        # v1-column for first partite set
        # v2-column for second partite set
        # w-column of weights
        # GraphName - name of the graph image and file to be saved
        
        
        # initialise the dataframe
        # dfsort=df1.groupby([v1,v2], as_index=False).unique() 
        
        
        vertex1 = list(df1[v1].unique())
        vertex2 = list(df1[v2].unique())
        # list of each half of the tuples that form the ends of an edge, and its weight
        tuple1=list(df1[v1])
        tuple2=list(df1[v2])
        tuple3=list(df1[w])
        #list of weighted edges
        
        weighted_edgelist=list(zip(tuple1,tuple2,tuple3))
        
        # build the graph
        G = nx.Graph()
        G.add_nodes_from(vertex1, bipartite=0) 
        G.add_nodes_from(vertex2, bipartite=1)
        # add all edges and weights
        G.add_weighted_edges_from(weighted_edgelist)
        
        # generate the Biadjacency matrix, put it into a dataframe, and save as excel
        
        Adjacency_matrix = bipartite.biadjacency_matrix(G,vertex1,vertex2)
        
        
        TaskAdj_df=pd.DataFrame(Adjacency_matrix.toarray())
        TaskAdj_df.index=vertex1
        TaskAdj_df.columns=vertex2
        
        # save dataframe to excel
        ExcelTitle=GraphName+'.xlsx'
        TaskAdj_df.to_excel(ExcelTitle,startrow=0, startcol=0)
            
        # pickle the graph
        GraphTitle=GraphName+'.gpickle'
        nx.write_gpickle(G, GraphTitle)
        
        # draw the graph
        # create positional dictinoary
        pos={}
        # place respective vertices into place
        pos.update((n, (1, i)) for i, n in enumerate(vertex1))
        pos.update((n, (2, i)) for i, n in enumerate(vertex2))
        nx.draw(G, pos=pos)
        Figname=GraphName+'.png'
        plt.savefig(Figname)
        plt.show()
    
        return G,Adjacency_matrix, TaskAdj_df
    
    # This function takes in a Biadjacency matrix and uses the projection found in
    # Mealy et al (2018) to construct the adjacency matrix
    @classmethod
    def Mealy_Projection(self, BiAdj_df, Selfloops=False, Directed=False):
        # BiAdj_df-the dataframe containing the biadjacency matrix
        # Selfloops - boolean for whether adjacency matrix is zero on the diagonal or not, default False    
        # Directed - boolean for whether graph is directed or undirected, default False
        
        
        # initalise row and column indicators
        Vertex1list = BiAdj_df.index.tolist()
        Vertex2list = BiAdj_df.columns.tolist()
        
        
        #1) generate a list of IWA to scarcity
        
        Scarcitylist=[]     
        # produce a list of task scarcity weights sw
        for i in Vertex2list:
            scarcity = 1/(BiAdj_df[i].sum())
            Scarcitylist.append(scarcity)
        
          
        # 2)transpose the BiAdjacency Matirx for each of calculation (pandas works along columns not rows)
        BiAdj_df = BiAdj_df.transpose()
            
        # 3) Produce ϕ(i,j) adjacency matrix
        
        
        # initalize the dataframe 
        zerolist = [0]*len(Vertex1list)
        df = pd.DataFrame(data=zerolist,index=Vertex1list, columns=['init col'])
          
        for i in Vertex1list:
          # generate a sequence of series ϕ=(ϕ(1),ϕ(2),...ϕ(n)) is a n*n matrix
          # where ϕ(i)=(ϕ(i,1),ϕ(i,2),...ϕ(i,n))' is a 1*n vector
          
          phi_list = []  
          for j in Vertex1list:
              
              # decide whether or not to include selfloops
              if Selfloops == False:    
                  if i == j:
                      phi = 0
                  else:
                      productSeries = BiAdj_df[i]*BiAdj_df[j]
                      list1 = list(productSeries)
                      weighted_product = [a*b for a,b in zip(list1,Scarcitylist)]
                      sum_weighted_product = sum(weighted_product)
                      
                      # Construct a directed or undirected matrix
                      if Directed == True:
                          phi = sum_weighted_product/sum(BiAdj_df[i])
                      else:
                          phi = min(sum_weighted_product/sum(BiAdj_df[i]), sum_weighted_product/sum(BiAdj_df[j]))
                  # this list is now ϕ(i)
                  phi_list.append(phi)
              else:
                  productSeries = BiAdj_df[i]*BiAdj_df[j]
                  list1 = list(productSeries)
                  weighted_product = [a*b for a,b in zip(list1,Scarcitylist)]
                  sum_weighted_product = sum(weighted_product)
                  
                  #Construct a directed or undirected matrix
                  if Directed == True:
                      phi = sum_weighted_product/sum(BiAdj_df[i])
                  else:
                      phi = min(sum_weighted_product/sum(BiAdj_df[i]), sum_weighted_product/sum(BiAdj_df[j]))
                  # this list is now ϕ(i)
                  phi_list.append(phi)
                  
                  
          # turn the list into a series    
          
          phi_series = pd.Series(phi_list)
          
          # once each series is generated, add it to the dataframe 
          df[i] = phi_series.values
        # clear the initalisation column      
        Ajd_df = df.drop('init col',axis=1)     
        
        
        return Ajd_df
    
class Spectral(object):
    
    def __init__(self,name):
        self.name=name
        
    
    
    # this function builds a graph out of any Adjacency Matrix, Spectrally Clusters it
    # and postions the nodes in the space spanned by the 2nd, 3rd and 4th largest eigenvalues
    # it also saves each nodes position alonge each dimension as a vector of values 1xn where
    # n is the number of nodes
    @classmethod
    def SpectralPlot(self, Adj_df, Figname,Directed=False):
        # Adj_df- dataframe of the adjacency matrix of the graph
        # Directed - Boolean for whether or not the graph is directed
        
        
        
        # get Adjacency Matrix as numpy array
        Adj_matrix = Adj_df.values
        
        # produce NetworkX Graph from Adjacency Matrix
        G = nx.from_numpy_matrix(Adj_matrix)
        
        # Plot two layouts of the graph, one spring weighted, the other a 2D spectral
        nx.draw_spectral(G)
        nx.draw_spring(G)
        
        
        # Generate the 3 dimensional spectral layout of G as a dictionary for each node
        spectral_dict = nx.spectral_layout(G, dim=3, weight='weight', scale=1)
        
        
        
        # construct x,y,z cordinate lists
        # initalise the lists
        
        x = []
        y = []
        z = []
        
        # get list of nodes: 
        nodelist=list(spectral_dict.keys())
        # for each node in the list, append x,y,z
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
    
    # save the output to a file:
    

    
    # This function will take a dataframe filled with a Graphs nodes and their attributes,
    # and produce a set of corrolations of those attributes with a certain number of eigenvectors
    
    @classmethod
    def SpectralCorr(self, df,Adj,c1,Attributelist,Numspec):
        # df - the dataframe containing the nodelist and their attributes
        # Adj - the adjacency matrix of the graph
        # c1 - the column containing the node identifiers in the df and Adj
        # Attributelist - the list of column headers from your dataframe to save as node attributes
        # Numspec - the interger number of dimensions of the spectral clustering
       
        # initalize enviroment: 
         
            
        # get dictionary of attribute vectors from dataframe 
        df.drop_duplicates(subset=c1,keep='first',inplace=True)
        nodelist = list(Adj.index)
            
        dict_att = dict()
        for att in Attributelist:
            list2=list(df[att])
            dict_att[att]=dict(zip(nodelist,list2))
            
        
        # construct cordinate lists in a dictionary for the spectral dimensions
        # initalise the dictionary
        dict_specdim = dict()
        for i in range(Numspec):
            dict_specdim[i] = []
        
         
        # build graph and spectrally cluster:
        
        # get Adjacency Matrix as numpy array
        Adj_matrix=Adj.values
        
        # produce NetworkX Graph from Adjacency Matrix
        G = nx.from_numpy_matrix(Adj_matrix)
        list1 = []
        for i in range(len(nodelist)):
            list1.append(i)
        node_dict = dict(zip(list1,nodelist))
        G=nx.relabel_nodes(G,node_dict)
        # Plot two layouts of the graph, one spring weighted, the other a 2D spectral
        nx.draw_spectral(G)
        nx.draw_spring(G)
        
        
        # Generate the 3 dimensional spectral layout of G as a dictionary for each node
        spectral_dict = nx.spectral_layout(G, dim=Numspec, weight='weight', scale=1)
        
        
        # get list of nodes: 
        
        
        # for each node in the list, append to each of the initalized lists that nodes position
        for i in range(Numspec):
            for node in nodelist:
                dict_specdim[i].append(spectral_dict.get(node).item(i))
            
        # produce the corralations      
        for att in dict_att:
            for specdim in dict_specdim:
                a=list(dict_att[att].values())
                b=list(dict_specdim[specdim])
                #come back and change
                
                # check about length
                print('alength:'+str(len(a)))
                print('blength:' + str(len(b)))
                           
                print('This is the corrolation matrix between '+ att + ' and spectal dimension '+str(specdim))
                print(scipy.stats.pearsonr(a, b))
                        
                # np.corrcoef(a,b))
          
        #print to a txt file    
        #with open('Spectral Corrolations.txt', 'w') as f:
         #   print('Spectral Corrolations.txt', )
    



    # This function finds the weighted Katz Centralilty for each node in
    # a Graph with Adjacency Matrix A
    
    @classmethod
    def Weighted_Katz(self, Adj,weight_vec):
        #A - Adjacency Matrix of Graph
        #alpha - scaling parameter
        #emplist - a vector to weight the Katz measure by
                 
        #Generate the right sized identty matrix
        Id = np.identity(Adj.shape)
        #note alpha should be smaller than 1/largest eigenvalue of A for traditional katz
        
        katz_mat =  np.linalg.inv(Id - 0.9*Adj)
            
        katz_weight_cent = katz_mat.dot(weight_vec)
        
        return katz_weight_cent
    
    # Take an Adjacency Matrix and Build its Normalized Laplacian """
    @classmethod
    def Laplacian(self,Adj):
        # Adj- Adjacency Matrix in a Dataframe format
        
        # Build graph
        G = nx.from_numpy_matrix(Adj.values)
        
        # Take Laplcian
        L =nx.laplacian_matrix(G).asfptype().todense()
         
        # getting node number
        node_num = len(G.nodes())
        
        # degree of each node
        deg = G.degree(weight='weight')
        deg = [deg[i] for i in range(node_num)]
        
        # construct diagonal of sqrt(degree)
        D_Half = np.diag(1./np.sqrt(deg))
        
        # construct normalized Laplacian
        first_dot = np.dot(D_Half,L)
        LapNorm = np.dot(D_Half,first_dot)
        
        return LapNorm


    #This is a function which takes the adjacency of a Graph,
    # and returns the orders list of eigenvectors of the Laplacian and runs k-means over them"""
    @classmethod
    def SpectralLabels(self,Adj,num_clus=10):
        #Adj - Graphs adjacency matrix
        #Number of clusters 
        
        # get Lapnorm
        LapNorm=Spectral.Laplacian(Adj)
        
        # generate the eigenvectors and values
        eigenValues, eigenVectors  = np.linalg.eig (LapNorm)
        
        # generate the list of ordering for eigenvectors
        idx = eigenValues.argsort()  
        eigenValues = eigenValues[idx]
        eigenVectors = eigenVectors[:,idx]
        
        # put them in appropriate data format
        eigenVectors = [eigenVectors[:, i] for i in range(len(eigenVectors))]
        eigenVectors = [np.squeeze(np.asarray(eigenVectors[i])) for i in range(len(eigenVectors))]
    
        # take the top n eigenvectors and cluster
        # build matrix of the top two
        A = np.column_stack((eigenVectors[1], eigenVectors[2]))
        # add in the rest
        for i in range(num_clus - 2):
            A = np.column_stack((A, eigenVectors[i + 3]))
    
        # Computing k-means
        kmeans = KMeans(n_clusters=num_clus)
        kmeans.fit(A)
        centroid = kmeans.cluster_centers_
        labels = kmeans.labels_
        
        return labels
    

    # This function takes the vector of clustering labels, and inserts it into a dataframe of 
    # node identifiers"""
    @classmethod
    def SpectralID(self,Adj,IDdf,m1,c1,numclus=10):
        # labels - the labeled vector of cluster identifiers
        # Adj - the original adjacency matrix
        # LapNorm - the Normalized Laplacian
        # IDdf - the dataframe containing the node identifiers
        # c1 - the column name in IDdf that contains node name
        # m1 - the column name in IDdf that contains the node ID found in Adj
        # numclus - the number of clusters you want to ID
        
        labels = Spectral.SpectralLabels(Adj, num_clus=numclus)
        
        # strip the adjacency matrix down to only one column and its index
        
        
        Adj1 = Adj.iloc[:,0]
        Adj1 = Adj1.reset_index()
        
        # add the LabelSeries
        
        Adj1['Cluster']=pd.Series(labels)
        
        # drop the irrelevant column
        
        Adj1 = Adj1[['index','Cluster']]
        
        # order Adj1 by cluster
        
        Adj2 = Adj1.sort_values('Cluster')
        
        Adj2.columns=[m1,'Cluster']
        
        # add in the node names from IDdf
        
        final_df = PandaFormat.df_single_map(IDdf,Adj2,m1,c1)
        
        return final_df
    
    
        # This function takes a normalized Laplacian of a graph and 
        # returns the Spectral Gap Diagram"""
    
    @classmethod
    def SpectralGap(self, Adj):
        #NormLap - The Normalized Laplacian
        
        LapNorm=Spectral.Laplacian(Adj)
        
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


    # This function builds a graph out of any Adjacency Matrix, Spectrally Clusters it
    # and postions the nodes in the space spanned by k largest eigenvalues and saves this n*k matrix as 
    # a dataframe.
    @classmethod
    def Kdim_SpectralPlot(self, Adj_df,k, Directed=False):
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
        
    # This function produces a dictionary for each Cluster dataframe that maps each Occ Code (key) to 
    # a cluster number (value) and inserts those cluster identifiers into a df"""
    @classmethod    
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

    # This function takes the mean and the standard deviation of the co-ordinates of occupations
    # in Spectral space of all occupations and also occuaptions within each cluster
    @classmethod
    def SpanAnalysis(self, Spandf,k,ClusterID):
        #Spandf - the dataframe containing the spectral co-ordinates of the occupations and ClusterID
        # k the number of Clusters
        # ClusterID - the column name containing the Cluster identifiers
        
        # loop over the k dimensions
        for i in range(k):
            
            # produce the mean and stdev of all occupations in each eigenvector
            series=pd.Series(Spandf[i])
            smean=series.mean()
            sstd=series.std()
            print('Mean Value in EigenVector ' +str(i) + ': ' + str(smean))
            print('Std in EigenVector ' +str(i) +': ' +str(sstd))
        
        # now calculate in each cluster
        for i in range(k): # the i's are the eigenvectors
            meanlist=[]
            stdlist=[]
            for j in range(k):# the j's are the clusters
                clus_series=Spandf.loc[Spandf[ClusterID]==j,i]
                clusmean=clus_series.mean()
                clus_std=clus_series.std()
                meanlist.append(clusmean)
                stdlist.append(clus_std)
                
            print('Cluster means '+str(i)+': ')
            print(meanlist)
            print('Cluster std'+str(i)+': ')
            print(stdlist)
            
        
    # This function takes the mean and the standard deviation of the log co-ordinates of occupations
    # in Spectral space of all occupations and also occuaptions within each cluster
    @classmethod
    def LogSpanAnalysis(self, Spandf,k,ClusterID):
        # Spandf - the dataframe containing the spectral co-ordinates of the occupations and ClusterID
        # k the number of Clusters
        # ClusterID - the column name containing the Cluster identifiers
        
        # get log co-ordinates
        Spandf.applymap(lambda x: abs(x))
        Spandf.applymap(lambda x: math.log(x))
        # loop over the k dimensions
        for i in range(k):
            
            
            # produce the mean and stdev of all occupations in each eigenvector
            series=pd.Series(Spandf[i])
            smean=series.mean()
            sstd=series.std()
            print('Mean Value in EigenVector ' +str(i) + ': ' + str(smean))
            print('Std in EigenVector ' +str(i) +': ' +str(sstd))
        
        # now calculate in each cluster
        for i in range(k): # the i's are the eigenvectors
            meanlist=[]
            stdlist=[]
            for j in range(k):# the j's are the clusters
                clus_series=Spandf.loc[Spandf[ClusterID]==j,i]
                clusmean=clus_series.mean()
                clus_std=clus_series.std()
                meanlist.append(clusmean)
                stdlist.append(clus_std)
                
            print('Cluster means '+str(i)+': ')
            print(meanlist)
            print('Cluster std'+str(i)+': ')
            print(stdlist)  