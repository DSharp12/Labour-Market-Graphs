"""This Code takes O*NET data on Tasks, DWA's and IWA's. It weights IWA's for each occupations
based on their component DWA and Task work intensities. It uses these ranks as weights in a
Bipartite Graph, building the graph and returning it and it's adjacency matrix.
""" 

#import relevant files

import numpy as np
import pandas as pd
import networkx as nx 
from networkx import bipartite
import matplotlib.pyplot as plt
import os

#set filepath
path=r"C:\Users\Daniel Sharp\Documents\MPhil Thesis\Data and Analysis"
os.chdir(path)


#define relevant functions, each will be called below

"""This function generates a sequence of lists of unique items from columns in a dataframe
"""
def unique_lists(df,collist):
    #df - dataframe you are generating lists from
    #collist - list of column names whose unique values you want to generate
    
    #run through each column title in a for loop
    for i in collist:
        nospace=i.replace(" ","")
        #create the name of the list
        listname=nospace+'_list'
        #create the list
        list1=list(df[i].unique())
        #change the variable to the listname
        exec(listname + "=  list(df[i].unique())")
        
        return list1
    


"""A function that generates a multi_index dictionary from a dataframe
"""
def df_multi_dict(df1,m1,m2,c1):
    #df1-dataframe 1
    #m1-column containing the first dictionary key
    #m2-column containing the second dictionary key
    #c1-column containing the dictionary values 
   
    #sort the dataframes to generate ordered lists
    df1=df1.sort_values([m1,m2])
    #generate a list of the first index
    m1list=list(df1[m1]) 
   #generate a list of the second index
    m2list=list(df1[m2])
   #ties indices together in a list of tuples 
    m1m2list=list(zip(m1list,m2list)) 
   #generate the list of values for tuple keys
    c1list=list(df1[c1])
   #create dictionary with tuple as key and column as values 
    df1_dict=dict(zip(m1m2list,c1list)) 
    return df1_dict

"""A function that maps from one dataframe to another, conditional on multiple coloumn values
"""    
def df_multi_map(df1,df2,m1,m2,c1):
    #df1-dataframe 1 - exporting column values
    #df2-dataframe 2 - importing column values
    #m1-first column from map 
    #m2-second (nested) column from map
    #c1-column containing the dictionary values 
   
    #sort values to get ordering
    df2=df2.sort_values([m1,m2])
    #generate the desired mapping from the first dataframe with a tuple-key dictionary
    df1_dict=df_multi_dict(df1,m1,m2,c1)  
    #generate the tuple key in the second dataframe
    m1list=list(df2[m1]) 
    m2list=list(df2[m2])
    df2['tuplelist']= list(zip(m1list,m2list))
    #generate the empty column to fill with called values
    df2[c1]="" 
    #for each tupple in the second dataframe, call the value from the dictionary   
    df2[c1]=df2['tuplelist'].map(df1_dict)    
    df2.drop('tuplelist',axis=1, inplace=True)
    #test for poor mapping
    print(df2.isna().sum())
    df2.dropna(inplace=True)
    return df2    

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

"""Test the functions
"""
#ratedftest=ratedf.head(100)
#dftest=df.head(100)
#testoutput=df_multi_map(ratedftest,dftest,'O*NET-SOC Code','Task ID','Data Value')
#testoutput_single=df_single_map(ratedftest,dftest,'O*NET-SOC Code','Task ID')

"""This function conducts a nested sort from a list on a GroupBy object
"""
def sort_grp(df, list1):
    #we take the dataframe, as sort along the columns
    df=df.sort_values(list1, ascending=[False,False,False])
    
    return df

    
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

"""This function normalizes nested values in a dataframe"""
def normal_nest(df,m1,m2,nestlist):
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
  

#drop any repetitions of IWA's within SOC's
"""This function drops duplicates of c1 within a GroupBy Object
"""
def within_drop(df,c1):
    df=df.drop_duplicates(subset=c1,keep='first')
    return df 


   
"""This function takes column elements that are lists, and extends the dataframe to include
them in their own column
"""
def list_extend(df,m1,m2,c1):
    #df - dataframe 
    #m1 - first group
    #m2 - second group
    #c1 - column containing lists
    
    #get the nested index to return m1,m2         
    x=df.index[0]
    #get m1,m2 to repeat in the right lengths 
    length=len(df.at[x,c1])
    #set global variables to check intermediary steps in the functions process
    global m1list
    #produce three lists, all of equal length, with each of m1,m2 repeated the size of the df
    m1list=[df.at[x,m1]]*length
    m2list=[df.at[x,m2]]*length
    
    c1list=list(df.at[x,c1])
    tuplelist=zip(m1list,m2list,c1list)
    #turn the zipped lists in  a list of lists
    list_of_tup = [list(elem) for elem in tuplelist]
    #then apply these lists to the three columns in df2
    df2 = pd.DataFrame(list_of_tup, columns=[m1,m2,c1])
    #add those three columns to the bottom of the df, to be stacked in 'groupby.apply'
    df=df.append(df2)
    df=df.iloc[1:]

    return df

"""This function takes the BiAdjacency binary matrix and returns a dataframe 
with a nested list, for each partite, the elements of the other partite associated with it
"""
def Biadj_todf(Adj, columnlist):
    #Adj - adjacency matrix
    #columnlist - list of column headers (each partite)
    
    #generate lists of nodes in each vertex
    vertex1list=list(Adj)
    #initalise dataframe
    df=pd.DataFrame(columns=columnlist)   
    
    #generate a dictionary, for each column value, of a dictionary of skills and 
    #binary value associated with it, the drop all the key,value pairs where value is 0
    meta_dict=Adj.to_dict()
    
    for element in vertex1list:
        #drop items where value is 0
        meta_dict[element]={ k:v for k, v in meta_dict[element].items() if v==1 }
        #get list of the nodes in vertex two associated with node: element
        v2sublist=list(meta_dict[element].keys())
        #add multiples of vertex1 to the dataframe 
        elementlist=[element]*len(v2sublist)
        tuplelist=zip(elementlist,v2sublist)
        list_of_tup = [list(elem) for elem in tuplelist]
        df2 = pd.DataFrame(list_of_tup, columns=columnlist)
        df=df.append(df2)
        
    return df


#construct the weights off the rank or datavalue of each IWA in each Occupation



#check to see any repetitions of IWA's in each SOC
#numiwa=finaldf.groupby(by=['O*NET-SOC Code', 'IWA ID'], as_index=False).count()[['O*NET-SOC Code','DWA ID', 'IWA ID']]
#numiwa.rename(columns={'DWA ID':'IWA Freq'},inplace=True)
 
#4) Build the weighted bipartite graph from the dataframe
"""This function takes a dataframe and builds a weighted bipartite graph"""

def df_bipartite(df1,v1,v2,w,GraphName):
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

#Begin the Data Analysis 





#1)using pandas to read in the data in excel format as dataframes
df = pd.read_excel('Tasks to DWAs.xlsx', header=0)
ratedf = pd.read_excel(r'C:\Users\Daniel Sharp\Documents\MPhil Thesis\Data and Analysis\Task Ratings.xlsx', header=0)
iwadf=pd.read_excel(r'C:\Users\Daniel Sharp\Documents\MPhil Thesis\Data and Analysis\DWA Reference.xlsx', header=0)



skilldf=pd.read_excel(r'C:\Users\Daniel Sharp\Documents\MPhil Thesis\Data and Analysis\Skills.xlsx', header=0)
abilitydf=pd.read_excel(r'C:\Users\Daniel Sharp\Documents\MPhil Thesis\Data and Analysis\Abilities.xlsx')






skilliwamatrix=pd.read_excel(r'C:\Users\Daniel Sharp\Documents\MPhil Thesis\Data and Analysis\Skills to IWA.xlsx',header=0)
skilliwamatrix.set_index('Element ID', inplace=True)
skilliwadf=Biadj_todf(skilliwamatrix,['IWA ID','Skill ID'])
abilityIWAmatrix=pd.read_excel(r'C:\Users\Daniel Sharp\Documents\MPhil Thesis\Data and Analysis\In_Ab.xlsx',header=0)
abilityiwadf=Biadj_todf(abilityIWAmatrix,['IWA ID','Ability ID'])


#2)produce lists of elements
#list of uniques occupations
occs = list(df['O*NET-SOC Code'].unique())
#list of unique tasks
tasks = list(df['Task ID'].unique())
#list of unique dwas 
dwaids = list(df['DWA ID'].unique())
#list of unique IWA's
iwas=list(iwadf['IWA ID'].unique())


#gives unique task codes list related to occupation
uniqtasks = df.groupby('O*NET-SOC Code')['Task ID'].unique().reset_index()
#gives unique DWA code list related to occupation
uniqdwa = df.groupby('O*NET-SOC Code')['DWA ID'].unique().reset_index()
#give number of unique tasks for each occ
numuniqtasks = df.groupby(by='O*NET-SOC Code', as_index=False).agg({'Task ID': pd.Series.nunique})


#3) Collect the list of all tasks associated with DWA's across occupations
dwas = df.groupby('DWA ID')['Task ID'].unique().reset_index()
#print(dwas)

#give number of unique dwas linked to each task
numdwas = df.groupby(by='DWA ID', as_index=False).agg({'Task ID': pd.Series.nunique})
#print(numdwas)    

#give the number of tasks for each DWA in each occupation
numtasks=df.groupby(by=['O*NET-SOC Code', 'DWA ID'], as_index=False).agg({'Task ID': pd.Series.nunique})
numtasks.rename(columns={'Task ID':'Task Freq'},inplace=True)

#reading in csv file of dwa rankings
ratedf = ratedf.loc[:, ['O*NET-SOC Code', 'Task ID', 'Scale ID', 'Data Value']]

#removing other ranking classifications from dataframe - FT = frequency and RT = relevance - only need IM = importance
ratedf = ratedf[ratedf['Scale ID'] != 'FT']
ratedf = ratedf[ratedf['Scale ID'] != 'RT']

skilllevel=skilldf[skilldf['Scale ID']=='LV']
skilllevel.rename(columns={'Element ID':'Skill ID'},inplace=True)
skillimp=skilldf[skilldf['Scale ID']=='IM']

abilitylevel=abilitydf[abilitydf['Scale ID']=='LV']
abilitylevel.rename(columns={'Element ID':'Ability ID'},inplace=True)
abilityimp=abilitydf[abilitydf['Scale ID']=='IM']

#print(ratedf)

#4)map the dataframes together and onstruct the weighting's to choose from


#4a) Generate a weighting based on the task intensity score

#create combined dataframe for DWA and Data Value
dfvalue=df_multi_map(ratedf,df,'O*NET-SOC Code','Task ID','Data Value')
#create another dataframe to add in IWA ID's
dfIWA=df_single_map(iwadf,dfvalue,'DWA ID','IWA ID')
#add on a coloumn for frequencey of task input into each DWA
dffreq=df_multi_map(numtasks,dfIWA,'O*NET-SOC Code','DWA ID', 'Task Freq')

#take the average data value and task freq for each IWA for each datavalue 
dfavg=average_nest(dffreq,'O*NET-SOC Code','IWA ID',['Data Value','Task Freq'])


#sort tasks with their associated IWA's by Average Data Value and Average Task Frequency for each Occupation
sortdf=dfavg.groupby('O*NET-SOC Code').apply(sort_grp, list1=['Average Data Value','Average Task Freq','IWA ID'])
sortdf.reset_index(drop=True,inplace=True) 

#drop any repetitions of IWA's within SOC's
uniIWAdf_task=sortdf.groupby('O*NET-SOC Code').apply(within_drop, c1='IWA ID')

#trim down the dataframe to only include relevant variables
finaldf_task=uniIWAdf_task[['O*NET-SOC Code', 'Task ID', 'DWA ID', 'IWA ID', 'Average Data Value', 'Average Task Freq']]
finaldf_task.reset_index(drop=True,inplace=True) 



#4b) Generate a weighting based on the average skill level of those IWAs in those occupations 

#first get a dataframe containnig the unique set of IWA per SOC
dfIWAtrim=dfIWA[['O*NET-SOC Code','IWA ID']].groupby('O*NET-SOC Code').apply(within_drop, c1='IWA ID')
dfIWAtrim.reset_index(drop=True, inplace=True)

#then get the dataframe containing, for each IWA, the list of associated Skills
iwa_skilllistdf=skilliwadf.groupby('IWA ID')['Skill ID'].unique().reset_index()
#map that list to each IWA within each SOC
SOC_IWA_SKILLdf=df_single_map(iwa_skilllistdf,dfIWAtrim,'IWA ID','Skill ID')

#set up the map from SOC to IWA to Skill
skillweightdf=SOC_IWA_SKILLdf.groupby(['O*NET-SOC Code','IWA ID']).apply(list_extend,m1='O*NET-SOC Code',m2='IWA ID',c1='Skill ID')
#map the weights from the SOC-Skill mapping to the SOC-IWA-Skill dataframe
skillweightdf2=df_multi_map(skilllevel,skillweightdf,'O*NET-SOC Code','Skill ID','Data Value')
skillweightdf2.sort_values(['O*NET-SOC Code','IWA ID'],inplace=True)
skillweightdf2.reset_index(drop=True,inplace=True)

#find the average skill level of each IWA, and reduce the dataframe size
avgiwaskill=average_nest(skillweightdf2,'O*NET-SOC Code','IWA ID',['Data Value'])
avgiwaskill.reset_index(drop=True,inplace=False)

uniIWAdf_skill=avgiwaskill.groupby('O*NET-SOC Code').apply(within_drop, c1='IWA ID')
uniIWAdf_skill.reset_index(drop=True,inplace=True)

#produce the final dataframe for IWA Skills weighting
finaldf_skill=uniIWAdf_skill[['O*NET-SOC Code', 'IWA ID', 'Average Data Value']]
finaldf_skill.reset_index(drop=True,inplace=True) 

"""
#4c) Generate a weighting based on the ability intensity score
#get the dataframe containing, for each IWA, the list of associated Abilities
iwa_abilitylistdf=abilityiwadf.groupby('IWA ID')['Ability ID'].unique().reset_index()
#map that list to each IWA within each SOC
SOC_IWA_ABILITYdf=df_single_map(abilityiwadf,dfIWAtrim,'IWA ID','Ability ID')

#set up the map from SOC to IWA to Ability
abilityweightdf=SOC_IWA_SKILLdf.groupby(['O*NET-SOC Code','Ability ID']).apply(list_extend,m1='O*NET-SOC Code',m2='IWA ID',c1='Skill ID')
#map the weights from the SOC-Ability mapping to the SOC-IWA-Ability dataframe
Abilityweightdf2=df_multi_map(skilllevel,abilityweightdf,'O*NET-SOC Code','Skill ID','Data Value')
Abilityweightdf2.sort_values(['O*NET-SOC Code','IWA ID'],inplace=True)
Abilityweightdf2.reset_index(drop=True,inplace=True)

#find the average skill level of each IWA, and reduce the dataframe size
avgiwaability=average_nest(Abilityweightdf2,'O*NET-SOC Code','IWA ID',['Ability Value'])
avgiwaability.reset_index(drop=True,inplace=False)

uniIWAdf_ability=avgiwaability.groupby('O*NET-SOC Code').apply(within_drop, c1='Ability ID')
uniIWAdf_ability.reset_index(drop=True,inplace=True)

#produce the final dataframe for IWA Skills weighting
finaldf_ability=uniIWAdf_ability[['O*NET-SOC Code', 'Ability ID', 'Average Data Value']]
finaldf_ability.reset_index(drop=True,inplace=True) """


#5)build the bipartite graphs
#build the weighted bipartite graphs, and their adjacency matrix
SOC_IWA_skill_bipartite=df_bipartite(finaldf_skill,'O*NET-SOC Code','IWA ID','Average Data Value','Bipartite_skill')
SkillAdj_df=TaskAdj_df
SOC_IWA_task_bipartite=df_bipartite(finaldf_task,'O*NET-SOC Code','IWA ID','Average Data Value','Bipartite_task')






