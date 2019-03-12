"""This code will take in adjacency matricies of graphs, and datasets of node attributes and 
examine the relationships between node attributes and graph structure
It will produce images of the graphs for illistrative and analytical purposes
"""

import numpy as np
import pandas as pd
import networkx as nx
from networkx import bipartite
from bokeh.io import show, output_notebook
from bokeh.plotting import figure
import os
import matplotlib.pyplot as plt
from bokeh.models import Range1d, Plot
from bokeh.models.graphs import from_networkx
from scipy import sparse


#set filepath
filepath =r'C:\Users\Daniel Sharp\OneDrive - Nexus365\MPhil Thesis\Data and Analysis\Adjacency Matrices\SOC'
os.chdir(filepath)

"""Function Enviroment
"""
def BiGraphViz(biadj):
    #biadj - the biadjacency matrix as a pandas dataframe
    
    #generate a networkx graph object
    nump=biadj.values
    scipy_adj=sparse.csr_matrix(nump)

    G=bipartite.from_biadjacency_matrix(scipy_adj)
    
    #set Bokeh enviroment
    
    #describe how you want it drawn
    graph = from_networkx(G, nx.drawing.layout._sparse_spectral)
    
    
    #describe how you want it plotted
    plot = Plot(x_range=Range1d(-2, 2), y_range=Range1d(-2, 2))
    plot.renderers.append(graph)
    graph.node_renderer.glyph.update(size=20, fill_color="orange")

    # Set some edge properties too
    graph.edge_renderer.glyph.line_dash = [2,2]
    
    #show the final output
    show(plot)
    
    
"""Run Enviroment
"""

#import the adjacency matrices you want to 
#graphs
adj1=pd.read_excel('Task_Adj_Matrix.xlsx')
#bipartite graphs
bipar1=pd.read_excel('SOC Unweighted Bipartite.xlsx')
bipar2=pd.read_excel('SOC Task Weighted Bipartite.xlsx')
    
    