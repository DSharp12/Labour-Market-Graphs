"""This code replicates Mealy et al (2018) New Lense on Labour Paper, it maps from DWA's to IWA's for 
SOC's and maps these SOC's to ACS codes. It takes an unweighted bipartite, and projects it onto the
occupation nodes. Then it looks at the corrolations of this matrix vs the other matrix. 
"""
#initalise enviroment
#import modules:
import pandas as pd
import os
import networkx as nx
from networkx import bipartite
import matplotlib.pyplot as plt
#set filepath
path=r"C:\Users\Daniel Sharp\OneDrive - Nexus365\MPhil Thesis\Data and Analysis"
os.chdir(path)

#Construct the mapping of SOC to ACS Codes

"""This function produces a mapper from a 8 didgit SOC to a 6 didgit SOC"""
def Code_mapper(df,c1,c2):
    #df1 -the dataframe containing c1 the granular codee
    #c1 - column title for granular code
    #c2 - column title for non granular code
    granularlist=list(df[c1])
    assert granularlist==list(df[c1])
    
    non_granularlist=[]
    for element in granularlist:
        #the 8 didgit SOC is split from the 6 digit by a '.'
        non_granularlist.append(element.split(".")[0])
    assert len(granularlist)==len(non_granularlist)
    df[c2]=non_granularlist
    
    return df


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


"""This function takes a mapping from column m1 to column c1 in df1, and pulls the elements of 
c1 to df2 according to that mapping, this is used to take the 6 digit SOC to a ACS code
"""
def df_single_map(df1,df2,m1,c1):
    #df1-dataframe 1 exporting column values
    #df2-dataframe 2 importing column values
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
                  phi=1
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

"""This function takes a dataframe and builds a unweighted bipartite graph
it then returns the graph, and saved the biadjacency matrix as a dataframe"""

def df_bipartite(df1,v1,v2):
    #df1-dataframe with partite sets as columns, and a column of weights
    #v1-column for first partite set
    #v2-column for second partite set
    #GraphName - name of the graph image and file to be saved
    
    
    #initialise the dataframe
    #dfsort=df1.groupby([v1,v2], as_index=False).unique() 
    
    global vertex1
    global vertex2
    vertex1=list(df1[v1].unique())
    vertex2=list(df1[v2].unique())
    #list of each half of the tuples that form the ends of an edge, and its weight
    tuple1=list(df1[v1])
    tuple2=list(df1[v2])
    
    #list of edges
    
    edgelist=list(zip(tuple1,tuple2))
    
    #build the graph
    G=nx.Graph()
    G.add_nodes_from(vertex1, bipartite=0) 
    G.add_nodes_from(vertex2, bipartite=1)
    #add all edges
    G.add_edges_from(edgelist)
    
    #generate the Biadjacency matrix, put it into a dataframe, and save as excel
    global Adjacency_matrix
    Adjacency_matrix=bipartite.biadjacency_matrix(G,vertex1,vertex2)
    
    global TaskAdj_df
    TaskAdj_df=pd.DataFrame(Adjacency_matrix.toarray())
    TaskAdj_df.index=vertex1
    TaskAdj_df.columns=vertex2
    
    return G              

"""This function takes a dataframe and builds a weighted bipartite graph"""

def df_weighted_bipartite(df1,v1,v2,w,GraphName):
    #df1-dataframe with partite sets as columns, and a column of weights
    #v1-column for first partite set
    #v2-column for second partite set
    #w-column of weights
    #GraphName - name of the graph image and file to be saved
    
    
    #initialise the dataframe
    #dfsort=df1.groupby([v1,v2], as_index=False).unique() 
    
    global vertex1
    global vertex2
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
    global Adjacency_matrix
    Adjacency_matrix=bipartite.biadjacency_matrix(G,vertex1,vertex2)
    
    global TaskAdj_df
    TaskAdj_df=pd.DataFrame(Adjacency_matrix.toarray())
    TaskAdj_df.index=vertex1
    TaskAdj_df.columns=vertex2
    
    #save dataframe to excel
    ExcelTitle=GraphName+'.xlsx'
    TaskAdj_df.to_excel(ExcelTitle,startrow=0, startcol=0)
        
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





def SpectralCorr(df,Adj,c1,Attributelist,Numspec):
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
    
   
    #label nodes in G with SOC codes 
    list1=[]
    for i in range(len(Adjorderlist)):
        list1.append(i)
    node_dict=dict(zip(list1,Adjorderlist))
    G=nx.relabel_nodes(G,node_dict)
    
    #assert np.testing.assert_array_equal(Adjorderlist, list(G.nodes))
      
    #1)generate top n (Numspec) eigenvectors using nx's spectral clustering function:
    #produces a dictionary of key: value pairs as node:[spectral co-ordinates in n dimensional space]
    spectral_dict=nx.spectral_layout(G, dim=Numspec, weight='weight', scale=1)
    
    
    #now we need to only keep those jobs that also appear in our df attribute data
    #generate a list of df node identifiers
    att_nodelist=list(df[c1])
    
    #new dictionary with only the keys in spectral_dict that show up in the df node's
    slim_specdict = {node: spectral_dict[node] for node in att_nodelist}
    
    
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
              eigendf[i].corr(eigendf[att])

"""This function takes the average value of multiple nested elements in a dataframe
   and adds them to the dataframe
"""
def average_nest(df,m1,m2,avglist):
    #df-dataframe
    #m1-fist column to group by
    #m2-second column to group by
    #avglist-list of columns you wish to average
    #avgnames-list of names of averge values column
    
    #initalise the new average columns for the dataframe
    df1=df #the function does not alter its original input dataframe
    for i in avglist:
        #find the mean of each list element 1 in each nested group
        dfmean=df1.groupby([m1,m2])[i].mean()
        #create a multi-index dictionary with the multi-index as key, average as value        
        list1=dfmean.index.tolist()
        list2=dfmean.tolist()
        avg_dict=dict(zip(list1,list2))
        #read the multi-index dictionary into the dataframe
        
        #generate the tuple list to map the average values
        m1list=list(df1[m1]) 
        m2list=list(df1[m2])
        df1['tuplelist']= pd.Series(list(zip(m1list,m2list)))
        
        
        df1.dropna(axis=0,subset=['tuplelist'],inplace=True)
        
        
        #pull average value column name from avgnames
        columnname='Average '+i 
        df1[columnname]=""
        #map average values from dictionary to dataframe   
        df1[columnname]=df1['tuplelist'].map(avg_dict)
    #clean up the dataframe    
    df1.drop('tuplelist',axis=1, inplace=True)
    
    return df1

"""Truncation function
"""
def f(x):
    if x <= 0.05: 
        return 0
    else: 
        return x

"""This function drops duplicates of c1 within a GroupBy Object
"""
def within_drop(df,c1):
    df=df.drop_duplicates(subset=c1,keep='first')
    return df 





#Load SOC-DWA mapping, SOC-ACS Mapping,DWA-IWA mapping,ACS-wage vector
SOCDWAdf= pd.read_excel('Tasks to DWAs.xlsx', header=0)
SOCACSdf=pd.read_excel('acs-crosswalk.xlsx',header=0)
IWADWAdf=pd.read_excel('DWA Reference.xlsx',header=0)
ACSwagedf=pd.read_csv('ACS_Properties.csv',header=0)


#apply the code mapper to the SOCDWA dataframe to generate the 6 didigt SOC's
mid_SOCDWAdf=Code_mapper(SOCDWAdf,'O*NET-SOC Code','SOC 2010 Code')

#map the ACS codes from SOCACSdf onto the SOCDWA dataframe using the single_map function
fin_SOCDWAdf=df_single_map(SOCACSdf,mid_SOCDWAdf,'SOC 2010 Code','ACS Code')

#map the IWA codes frrom the DWA IWA mapping to ACS-DWA mapping
finACSIWAdf=df_single_map(IWADWAdf,fin_SOCDWAdf,'DWA ID','IWA ID')

#take only the final bipartite between ACS codes and IWA codes
finaldf=finACSIWAdf[['ACS Code','IWA ID']]
finalSOCdf=finACSIWAdf[['O*NET-SOC Code','IWA ID']]


#generate a Biadjacency matrix from the final dataframe
Mealy_bipartite=df_bipartite(finaldf,'ACS Code','IWA ID')
Mealy_bipartite=df_bipartite(finalSOCdf,'O*NET-SOC Code','IWA ID')

TaskAdj_df.to_excel('ACS Unweighted Bipartite.xlsx')
UnweightedAdj=TaskAdj_df



#project the bipartite graph using Mealy Projection
JobSpaceAdj=Mealy_Projection(UnweightedAdj)

#get the ACS median log-wage vector
ACSwage=ACSwagedf[['acs_occ_code','log_median_earnings']]

#check the Spectral Corrolations 
SpectralCorr(ACSwage,JobSpaceAdj,'acs_occ_code','log_median_earnings',10)




"""This section of the code will extend Mealy et al(2018) work to weight the initial bipartite graph 
using our skill measure and the task intesnities
"""

#we load the dataframe containing the weighted SOC-IWA maps for skill weighted
#and task weighted bipartite graphs
SOC_IWA_Intensitydf=pd.read_excel('SOC_IWA Task Intensity.xlsx')
SOC_SkillIWA_Intensity=pd.read_excel('SOCSkillIWAdf.xlsx')



#get IWA intensity for each ACS code by merging the dataframes
#for tasks
mid_SOCTaskIntensitydf=Code_mapper(SOC_IWA_Intensitydf,'O*NET-SOC Code','SOC 2010 Code')
fin_SOCTaskIntensitydf=df_single_map(SOCACSdf,mid_SOCTaskIntensitydf,'SOC 2010 Code','ACS Code')

#for skill
mid_SOCSkillIntensitydf=Code_mapper(SOC_SkillIWA_Intensity,'O*NET-SOC Code','SOC 2010 Code')
fin_SOCSkillIntensitydf=df_single_map(SOCACSdf,mid_SOCSkillIntensitydf,'SOC 2010 Code','ACS Code')

#average the IWA intensity for all the SOC's within each ACS code to get unique
#IWA to ACS code intensity mapping

#for tasks
ACSIWAIntensity=average_nest(fin_SOCTaskIntensitydf, 'ACS Code','IWA ID',['Average Data Value'] )

#for skill
ACSSkillIntensity=average_nest(fin_SOCSkillIntensitydf,'ACS Code','IWA ID',['Average Data Value'])
#drop repeats of IWA's within ACS codes

#for tasks
FinACSIWAdf=ACSIWAIntensity.groupby('ACS Code').apply(within_drop, c1='IWA ID')

#for skill
FinACSSkilldf=ACSSkillIntensity.groupby('ACS Code').apply(within_drop, c1='IWA ID')

#pare down the dataframe ready for the bipartite graph
FinACSIWAdf=FinACSIWAdf[['IWA ID','ACS Code','Average Average Data Value']]
FinACSSkilldf=FinACSSkilldf[['IWA ID','ACS Code','Average Average Data Value']]


#build the ACS bipartite for tasks and skills

#for tasks
ACSTaskBipartite=df_weighted_bipartite(FinACSIWAdf,'ACS Code','IWA ID','Average Average Data Value','ACSIWATaskIntensityBipartite')
#save Bipartite Adjacency
TaskAdj_df.to_excel('ACS IWA Task Intensity Weighted Bipartite.xlsx')

#for skills
ACSSkillBipartite=df_weighted_bipartite(FinACSSkilldf,'ACS Code','IWA ID','Average Average Data Value','ACSIWATaskIntensityBipartite')
SkillAdj_df=TaskAdj_df

#save Bipartite Adjacency
SkillAdj_df.to_excel('ACS Skill weighted IWA Bipartite.xlsx')

#build the two ACS graphs

TaskWeightedJobSpace=Mealy_Projection(TaskAdj_df)
SkillWeightedJobSpace=Mealy_Projection(SkillAdj_df)

#save the projections
TaskWeightedJobSpace.to_excel('ACS Task Weighted Job Space.xlsx')
SkillWeightedJobSpace.to_excel('ACS Skill Weighted Task Job Space.xlsx')
JobSpaceAdj.to_excel('ACS Unweighted Job Space.xlsx')




#saivng the graphs
#mariatruncdf=mariadf.applymap(f)
#mariatruncdf.to_csv('mariatruncdf.csv')
#JobSpaceAdj.to_excel('UnweightedGraph.xlsx')



