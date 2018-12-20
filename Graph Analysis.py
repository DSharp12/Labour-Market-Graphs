"""This code will take in adjacency matricies of graphs, and datasets of node attributes and 
examine the relationships between node attributes and graph structure
"""

import pandas as pd
import numpy as np
import networkx as nx

#import the adjacency matrices you want to 
adj1=pd.read_excel(r'C:\Users\Daniel Sharp\Documents\MPhil Thesis\Data and Analysis\Task_Adj_matrix.xlsx',header=0)
adj2=pd.read_excel(r'C:\Users\Daniel Sharp\Documents\MPhil Thesis\Data and Analysis\Skill_Adj_matrix.xlsx',header=0)

"""This function takes an adjeceny matrix as a dataframe and normalises the  by the maximum
weight, so they to be between 0 and 1
"""
def max_normalize(Adj_matrix):
    #Adj_matrix - the Adjacency matrix to normalize
    
    