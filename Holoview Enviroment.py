"""This enviroment takes different datasets and uses Holoview to produce interactive enviroments of 
the datasets
"""

#import enviroment pre-requistites 
import numpy as np
import pandas as pd
import holoviews as hv
import networkx as nx
hv.extension('matplotlib')
hv.extension('bokeh')

%opts Graph [width=400 height=400]
 