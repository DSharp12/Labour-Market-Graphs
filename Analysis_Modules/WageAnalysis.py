# -*- coding: utf-8 -*-
"""
Created on Wed Mar  6 15:58:41 2019
This module is a sequence of code that is used for Wage and Employment Data Analysis
@author: Daniel Sharp
"""
import pandas as pd
import numpy as np
from scipy import stats
import seaborn as sns

class WageFormat(object):
    
    def __init__(self,name):
        self.name = name
        
    @classmethod
    def Code_mapper(self, df,c1,c2):
    # df1 -the dataframe containing c1 the granular codee
    # c1 - column title for granular code
    # c2 - column title for non granular code
    
        granularlist = list(df[c1])
        
        non_granularlist = []
        for element in granularlist:
            
            # the 8 didgit SOC is split from the 6 digit by a '.'
            non_granularlist.append(element.split(".")[0])
        
        df[c2] = non_granularlist
        
        return df  

    # This function holds certain columns, and renames the columns in a list of dataframes
    @classmethod
    def ColParse(self,dflist,dictt):
        # dflist - list of dataframes to rename
        # dictt - renaming dictionary
        # kieylist - list of keys to rename dataframes by
        # NOTE all dataframes in dflist must have dictt column name 
        
        
        # rename the dataframes
        for i, k in enumerate(dflist):
            dflist[i] = dflist[i][list(dictt.keys())]
        
        # cut down the dataframes
        for i, k in enumerate(dflist):
            dflist[i].rename(dictt,axis=1,inplace=True)
            # dflist[i].columns = changelist
        
        
        return dflist

    # This function returns a list of the mean value of clusters from a Pandas dataframe
    # it also returns the total average per year for comparison

    @classmethod
    def ClusterMean(self, df,clusterid,meanlist):
        # df - dataframe containing the values to be averaged 
        # clusterid -the columnid containing cluster identfiers
        # meanlist - the list of column values to be averaged
        
    
    
        # initalise the new average columns for the dataframe
        df1 = df.copy() # the function does not alter its original input dataframe
        # intialize dataframe to get the wage data
        dfinit = pd.DataFrame(data=[0]*len(df1[clusterid].unique()),index=df1[clusterid].unique(),columns=['init col'])
    
    
        # initalise a list to take the mean in each time period
        whole_meanlist = []
        
        for i in meanlist:
            
            # find the mean of each list element 1 in each nested group
            dfmean = df1.groupby(clusterid)[i].mean()
            
            dfinit[i] = dfmean
            # get mean of whole sample
            yearvar = df1[i].mean() 
            whole_meanlist.append(yearvar)
        
        dfinit=dfinit.transpose()
        # adjust column headings and add in whole mean
        firstcollist = list(dfinit.columns)
        secondcollist = ['Cluster: '+str(i) for i in firstcollist]
        dfinit.columns = secondcollist
        
        dfinit = dfinit.drop(labels='init col',axis=0)
        dfinit['Whole Sample'] = whole_meanlist  
        # add in year indicator
        dfinit['Year'] = [i for i in range(1999,2018)]
        
        return dfinit

    # This function cleans data of missing values and ensures 
    # that it is numeric in numeric columns
    @classmethod
    def DataCleaner(self, df,numericlist):
        
        for i in numericlist:
            df = df[df[i] != '***']
            df = df[df[i] != '*']
            df[i] = pd.to_numeric(df[i])
    
        return df

    # This function returns a list of the mean value of clusters from a Pandas dataframe
    @classmethod    
    def ClusterVar(self, df,clusterid,meanlist):
        # df - dataframe containing the mean values 
        # clusterid -the columnid containing cluster identfiers
        # meanlist - the list of column values to be averaged
        
        
        # initalise the new average columns for the dataframe
        df1 = df.copy() #the function does not alter its original input dataframe
        # intialize dataframe to get the wage data
        dfinit = pd.DataFrame(data=[0]*len(df1[clusterid].unique()),index=df1[clusterid].unique(),columns=['init col'])
        
        # initalise a list to take the mean in each time period
        whole_meanlist = []
        
        for i in meanlist:
            
            # find the mean of each list element 1 in each nested group
            dfstd = df1.groupby(clusterid)[i].std()
            
            dfinit[i] = dfstd
            
            yearvar = df1[i].std() 
            whole_meanlist.append(yearvar)
            
        dfinit = dfinit.transpose()
        firstcollist = list(dfinit.columns)
        secondcollist = ['Cluster: '+str(i) for i in firstcollist]
        dfinit.columns = secondcollist
        
        
        dfinit = dfinit.drop(labels='init col',axis=0)
        dfinit['Whole Sample'] = whole_meanlist
        # add year column
        dfinit['Year'] = [i for i in range(1999,2018)]
        
        return dfinit
    
    @classmethod
    def dfmerge(self,dflist, indexcol,namelist,mergecol):
    #dflist - list of dataframes you wish to merge
    #NOTE: all dataframes must have the same index colum with the same header
    
    
        indexlist=[list(dflist[i][indexcol]) for i,k in enumerate(dflist)]
        
        #create the intersection of the indices to ensure there is not merge issues
        setlist=[set(i) for i in indexlist]
        #run a for loop generating the intersection sets
        J=setlist[0].intersection(setlist[1])
        for i in setlist:
            J=J.intersection(i)
            
        #initalise a dataframe using merged index
        Mergeddf=pd.DataFrame(data=list(J),columns=['Indexcol'])
        
        #iterate through the columns in each of the dataframes
        for i,k in enumerate(dflist):
            
            name=str(namelist[i])
            
            col_dict=dict(zip(list(dflist[i][indexcol]),list(dflist[i][mergecol])))
            
            Mergeddf[name] = Mergeddf['Indexcol'].map(col_dict)
            
                  
        Mergeddf.rename({'Indexcol':indexcol},axis=1,inplace=True)    
                    
                
        return Mergeddf
    
    @classmethod
    def dfapplylog(self,df1, applylist):
        df=df1.copy()
        
        for i in applylist:
            ilist = [float(i) for i in df1[i]]
            df[i]=np.log(ilist)
        return df
       
    
    # This function takes two vectors of data and calculates the
    # Kolmogorov-Smirnov test on their Emperical Distribution Functions

class Statstest(object):
    
    def __init__(self, name):
        self.name = name
    
    @classmethod
    def KolSmir(self, x1,x2):
        # x1 - first vector of data
        # x2 - second vector of data
    
        v1=np.array(x1)
        v2=np.array(x2)
        
        return stats.ks_2samp(v1,v2)

     # This function takes a vector of data x and calculates the kde using seaborns distplot
     # it then produces a sequence of probability density's for 128 bins

    @classmethod
    def PDF_seaborn(x):
        # x - datavector you want the pdf readouts for
        
        densitylist = sns.distplot(x).get_lines()[0].get_data()[1]
        binwidth = 1/sum(densitylist)
        PDFlist = [i*binwidth for i in densitylist]
        # test for precision
        assert abs(1-sum(PDFlist)) < 0.0001
        
        return PDFlist

     # This function uses seaborns distplot to produce kernal density estimates 
     # and compares the entropy between those two plots specifically KL(x1||x2)

    @classmethod
    def KL_seaborn(pdfx1,pdfx2):
        # x1 - first vector of data
        # x2 - second vector of data
        # calculate the KL divergence of the two 
        arr1 = np.array(pdfx1)
        arr2 = np.array(pdfx2)
        
        return stats.entropy(arr1,arr2)


    # This function caclucates the change in the 90/10,90/50,50/10 percentiles 
    # of the wage distribution and outputs them as a list

    @classmethod
    def Diststatschange(self, vec1,vec2):
        # vec1 - newer vector of wages 
        # vec2 - older vector of wages
        
        ninetytenchange = ((np.percentile(vec1,90))/(np.percentile(vec1,10)))/((np.percentile(vec2,90))/(np.percentile(vec2,10)))
        ninetyfiftychange = ((np.percentile(vec1,90))/(np.percentile(vec1,50)))/((np.percentile(vec2,90))/(np.percentile(vec2,50)))
        fiftytenchange = ((np.percentile(vec1,50))/(np.percentile(vec1,10)))/((np.percentile(vec2,50))/(np.percentile(vec2,10)))
        
        return ninetytenchange,ninetyfiftychange,fiftytenchange

    # This function takes a series of different datasets and runs emperical tests on there wage distributions, and outputs
    # the results of these tests to a dataframe

    @classmethod
    def Disttests(self, arrlist1,arrlist2,cluslist,savename):
        # arrlist1 - lsit of np arrays of wages to test - Later in time
        # arrlist2 - list of np arras of wages to test - Earlier in time
        # savename - name of excel file to save to
        # initalise df
        dfcols = ['Cluster','Kol-Smir','90/10','90/50','50/10']
        df = pd.DataFrame(columns=dfcols)
        # produce naming list
        clus = cluslist
        # intialize lists
        KolSmirlist = []
        nineten = []
        ninefive = []
        fiveten = []
        
        # loop through dataframe lists
        for i,k in list(enumerate(arrlist1)): 
            for j,g in list(enumerate(arrlist2)):
                if i == j:
                    KS = Statstest.KolSmir(k,g)
                    KolSmirlist.append(KS)
                    # produce the 90/10,90/50,50/10 truple output 
                    x1,x2,x3 = Statstest.Diststatschange(k,g) #ensure dflist1 is the 'newer' wage dataframe
                    nineten.append(x1)
                    ninefive.append(x2)
                    fiveten.append(x3)
                    
        # test to ensure code is working
        assert len(KolSmirlist)==len(arrlist1)
        assert len(nineten)==len(arrlist1)
        
        # build dataframe
        df['Cluster'] = clus
        df['Kol-Smir'] = KolSmirlist
        df['90/10'] = nineten
        df['90/50'] = ninefive
        df['50/10'] = fiveten
        
        Savename = savename+'.xlsx'
        df.to_excel(Savename)
        return df
               
      
