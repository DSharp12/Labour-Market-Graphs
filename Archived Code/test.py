# -*- coding: utf-8 -*-
"""
Created on Mon Dec 17 10:27:44 2018

@author: Daniel Sharp
"""

abilityIWAmatrix=pd.read_excel(r'C:\Users\Daniel Sharp\Documents\MPhil Thesis\Data and Analysis\In_Ab.xlsx',header=0)
abilityiwadf=Biadj_todf(abilityIWAmatrix,['IWA ID','Ability ID'])
