# -*- coding: utf-8 -*-
"""
Created on Wed Feb 20 17:58:46 2019
This code takes the excel output from the ClusterID function, and produces analysis of the properties of these clusters
@author: Daniel Sharp
"""
import pandas as pd
import numpy as np
import seaborn as sns
import os
import matplotlib.pyplot as plt
import math
from scipy import stats
from scipy.ndimage.filters import gaussian_filter
from scipy.interpolate import make_interp_spline, BSpline
from scipy.interpolate import interp1d
import statsmodels.api as sm


#set filepath
path=r"C:\Users\Daniel Sharp\OneDrive - Nexus365\MPhil Thesis\Data and Analysis"
os.chdir(path)

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



"""This function takes a list of indexed dataframes and merges them into one single dataframe"""
def dfmerge(dflist, indexcol,namelist,mergecol):
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


                
"""This function produces a mapper from a refined SOC ID code to a less refined SOC ID code
"""
def Code_mapper(df,c1,c2):
    #df1 -the dataframe containing c1 the granular codee
    #c1 - column title for granular code
    #c2 - column title for non granular code
    granularlist=list(df[c1])
    non_granularlist=[]
    for element in granularlist:
        #the 8 didgit SOC is split from the 6 digit by a '.'
        non_granularlist.append(element.split(".")[0])
    df[c2]=non_granularlist
    
    return df              
  
               
"""This function holds certain columns, and renames the columns in a list of dataframes"""
def ColParse(dflist,dictt):
    #dflist - list of dataframes to rename
    #dictt - renaming dictionary
    #kieylist - list of keys to rename dataframes by
    #NOTE all dataframes in dflist must have dictt column name 
    
    
    #rename the dataframes
    for i, k in enumerate(dflist):
        dflist[i] = dflist[i][list(dictt.keys())]
    
    #cut down the dataframes
    for i, k in enumerate(dflist):
        dflist[i].rename(dictt,axis=1,inplace=True)
        #dflist[i].columns = changelist
    
    
    return dflist

"""This function returns a list of the mean value of clusters from a Pandas dataframe
    it also returns the total average per year for comparison
"""
def ClusterMean(df,clusterid,meanlist):
    #df - dataframe containing the values to be averaged 
    #clusterid -the columnid containing cluster identfiers
    #meanlist - the list of column values to be averaged
    


    #initalise the new average columns for the dataframe
    df1=df.copy() #the function does not alter its original input dataframe
    #intialize dataframe to get the wage data
    dfinit=pd.DataFrame(data=[0]*len(df1[clusterid].unique()),index=df1[clusterid].unique(),columns=['init col'])


    #initalise a list to take the mean in each time period
    whole_meanlist=[]
    
    for i in meanlist:
        
        #find the mean of each list element 1 in each nested group
        dfmean=df1.groupby(clusterid)[i].mean()
        
        dfinit[i]=dfmean
        #get mean of whole sample
        yearvar=df1[i].mean() 
        whole_meanlist.append(yearvar)
    
    dfinit=dfinit.transpose()
    #adjust column headings and add in whole mean
    firstcollist=list(dfinit.columns)
    secondcollist=['Cluster: '+str(i) for i in firstcollist]
    dfinit.columns=secondcollist
    
    dfinit=dfinit.drop(labels='init col',axis=0)
    dfinit['Whole Sample']=whole_meanlist  
    #add in year indicator
    dfinit['Year']=[i for i in range(1999,2018)]
    
    return dfinit

"""This function cleans data of missing values and ensures that it is numeric in numeric columns
"""
def DataCleaner(df,numericlist):
    
    for i in numericlist:
        df=df[df[i] != '***']
        df=df[df[i] != '*']
        df[i]=pd.to_numeric(df[i])

    return df

"""This function returns a list of the mean value of clusters from a Pandas dataframe"""
def ClusterVar(df,clusterid,meanlist):
    #df - dataframe containing the mean values 
    #clusterid -the columnid containing cluster identfiers
    #meanlist - the list of column values to be averaged
    
    
    #initalise the new average columns for the dataframe
    df1=df.copy() #the function does not alter its original input dataframe
    #intialize dataframe to get the wage data
    dfinit=pd.DataFrame(data=[0]*len(df1[clusterid].unique()),index=df1[clusterid].unique(),columns=['init col'])
    
    #initalise a list to take the mean in each time period
    whole_meanlist=[]
    
    for i in meanlist:
        
        #find the mean of each list element 1 in each nested group
        dfstd=df1.groupby(clusterid)[i].std()
        
        dfinit[i]=dfstd
        
        yearvar=df1[i].std() 
        whole_meanlist.append(yearvar)
        
    dfinit=dfinit.transpose()
    firstcollist=list(dfinit.columns)
    secondcollist=['Cluster: '+str(i) for i in firstcollist]
    dfinit.columns=secondcollist
    
    
    dfinit=dfinit.drop(labels='init col',axis=0)
    dfinit['Whole Sample']=whole_meanlist
    #add year column
    dfinit['Year']=[i for i in range(1999,2018)]
    
    return dfinit

"""This function takes two vectors of data and calculates the Kolmogorov-Smirnov test on their
Emperical Distribution Functions
"""

def KolSmir(x1,x2):
    #x1 - first vector of data
    #x2 - second vector of data

    v1=np.array(x1)
    v2=np.array(x2)
    
    return stats.ks_2samp(v1,v2)

"""This function takes a vector of data x and calculates the kde using seaborns distplot
it then produces a sequence of probability density's for 128 bins"""

def PDF_seaborn(x):
    #x - datavector you want the pdf readouts for
    
    densitylist=sns.distplot(x).get_lines()[0].get_data()[1]
    binwidth=1/sum(densitylist)
    PDFlist=[i*binwidth for i in densitylist]

    assert abs(1-sum(PDFlist)) < 0.0001
    
    return PDFlist

"""Calculate Distribution Statististics"""
"""This function uses seaborns distplot to produce kernal density estimates and compares the
entropy between those two plots specifically KL(x1||x2)
"""
def KL_seaborn(pdfx1,pdfx2):
    #x1 - first vector of data
    #x2 - second vector of data
    #calculate the KL divergence of the two 
    arr1=np.array(pdfx1)
    arr2=np.array(pdfx2)
    return stats.entropy(arr1,arr2)


"""This function caclucates the change in the 90/10,90/50,50/10 percentiles of the wage distribution 
and outputs them as a list
"""
def Diststatschange(vec1,vec2):
    #vec1 - newer vector of wages 
    #vec2 - older vector of wages
    
    ninetytenchange=((np.percentile(vec1,90))/(np.percentile(vec1,10)))/((np.percentile(vec2,90))/(np.percentile(vec2,10)))
    ninetyfiftychange=((np.percentile(vec1,90))/(np.percentile(vec1,50)))/((np.percentile(vec2,90))/(np.percentile(vec2,50)))
    fiftytenchange=((np.percentile(vec1,50))/(np.percentile(vec1,10)))/((np.percentile(vec2,50))/(np.percentile(vec2,10)))
    
    return ninetytenchange,ninetyfiftychange,fiftytenchange

"""This function takes a series of different datasets and runs emperical tests on there wage distributions, and outputs
the results of these tests to a dataframe
"""
def Disttests(arrlist1,arrlist2,cluslist,savename):
    #arrlist1 - lsit of np arrays of wages to test - Later in time
    #arrlist2 - list of np arras of wages to test - Earlier in time
    #savename - name of excel file to save to
    #initalise df
    dfcols=['Cluster','Kol-Smir','90/10','90/50','50/10']
    df=pd.DataFrame(columns=dfcols)
    #produce naming list
    clus=cluslist
    #intialize lists
    KolSmirlist=[]
    nineten=[]
    ninefive=[]
    fiveten=[]
    
    #loop through dataframe lists
    for i,k in list(enumerate(arrlist1)): 
        for j,g in list(enumerate(arrlist2)):
            if i ==j:
                KS=KolSmir(k,g)
                KolSmirlist.append(KS)
                #produce the 90/10,90/50,50/10 truple output 
                x1,x2,x3=Diststatschange(k,g) #ensure dflist1 is the 'newer' wage dataframe
                nineten.append(x1)
                ninefive.append(x2)
                fiveten.append(x3)
                
    #test to ensure code is working
    assert len(KolSmirlist)==len(arrlist1)
    assert len(nineten)==len(arrlist1)
    
    #build dataframe
    df['Cluster']=clus
    df['Kol-Smir']=KolSmirlist
    df['90/10']=nineten
    df['90/50']=ninefive
    df['50/10']=fiveten
    
    Savename=savename+'.xlsx'
    df.to_excel(Savename)
    return df

"""This function takes in data from a dataframe and plots the columns in a line graph next to 
each other
"""
def Pandaplt(df,xcol,ycollist,mark='',lwdth=2):
    #df - pandas dataframe containing the data for plotting
    #xol - column containing the x axis variable
    #yxollist - list of columns containing y variables for comparison 
    #mark - choice of marker
    colorlist=['b','g','r','c','m','y','k','brown','indigo','palegoldenrod','turquoise','peru','lawngreen']
    #test for enough colours
    assert len(colorlist) > len(ycollist)
    
    for i,k in enumerate(ycollist):
        plt.plot(xcol,k,data=df,marker=mark,color=colorlist[i],linewidth=lwdth)
    plt.legend()

def ClusterSum(df,clusterid,meanlist):
    #df - dataframe containing the values to be averaged 
    #clusterid -the columnid containing cluster identfiers
    #meanlist - the list of column values to be averaged
    


    #initalise the new average columns for the dataframe
    df1=df.copy() #the function does not alter its original input dataframe
    #intialize dataframe to get the wage data
    dfinit=pd.DataFrame(data=[0]*len(df1[clusterid].unique()),index=df1[clusterid].unique(),columns=['init col'])
    
    #initalise a list to take the mean in each time period
    whole_meanlist=[]
    
    for i in meanlist:
        
        #find the mean of each list element 1 in each nested group
        dfmean=df1.groupby(clusterid)[i].sum()
        
        dfinit[i]=dfmean
        #get mean of whole sample
        yearvar=df1[i].mean() 
        whole_meanlist.append(yearvar)
    
    dfinit=dfinit.transpose()
    #adjust column headings and add in whole mean
    firstcollist=list(dfinit.columns)
    secondcollist=['Cluster: '+str(i) for i in firstcollist]
    dfinit.columns=secondcollist
    
    dfinit=dfinit.drop(labels='init col',axis=0)
    dfinit['Whole Sample']=whole_meanlist  
    #add in year indicator
    dfinit['Year']=[i for i in range(1999,2018)]
    
    return dfinit
 
def Normalisedf(df1):
    df=df1.copy()
    dfcollist=list(df.columns)
    #intialise new df
    newdf=pd.DataFrame(index=df.index,columns=df.columns)
    for n, col in enumerate(dfcollist):
        normlist=[i/(df.iloc[0,n]) for i in df[col]]
        
        newdf[col]=normlist
    
    return newdf

def dfapplylog(df1, applylist):
    df=df1.copy()
    
    for i in applylist:
        ilist = [float(i) for i in df1[i]]
        df[i]=np.log(ilist)
    return df
    
"""Run Enviroment"""
#load dataframes
Task_clus=pd.read_excel('TaskweightedClusters.xlsx')
Un_clus=pd.read_excel('UnweightedClusters.xlsx')
Wagedf=pd.read_excel('2017BLSWage.xlsx')

#load wage and employment dataframes
wagedf_2017=pd.read_excel('2017BLSWage.xlsx')
wagedf_2016=pd.read_excel('2016BLSWage.xlsx')
wagedf_2015=pd.read_excel('2015BLSWage.xlsx')
wagedf_2014=pd.read_excel('2014BLSWage.xlsx')
wagedf_2013=pd.read_excel('2013BLSWage.xls')
wagedf_2012=pd.read_excel('2012BLSWage.xls')
wagedf_2011=pd.read_excel('2011BLSWage.xls')
wagedf_2010=pd.read_excel('2010BLSWage.xls')
wagedf_2009=pd.read_excel('2009BLSWage.xls')
wagedf_2008=pd.read_excel('2008BLSWage.xls')
wagedf_2007=pd.read_excel('2007BLSWage.xls')
wagedf_2006=pd.read_excel('2006BLSWage.xls')
wagedf_2005=pd.read_excel('2005BLSWage.xls')
wagedf_2004=pd.read_excel('2004BLSWage.xls')
wagedf_2003=pd.read_excel('2003BLSWage.xls')
wagedf_2002=pd.read_excel('2002BLSWage.xls')
wagedf_2001=pd.read_excel('2001BLSWage.xls')
wagedf_2000=pd.read_excel('2000BLSWage.xls')
wagedf_1999=pd.read_excel('1999BLSWage.xls')

#clean the wage dataframes
#Produce 6 digit SOC codes for wage mapping
Task_clus=Code_mapper(Task_clus,'O*NET-SOC Code','OCC_CODE')
Un_clus=Code_mapper(Un_clus,'O*NET-SOC Code','OCC_CODE')
#drop the 8 digit codes, and rename the 6 digit codes
Task_clus.drop(['O*NET-SOC Code'], axis=1,inplace=True)
Un_clus.drop(['O*NET-SOC Code'], axis=1,inplace=True)
Task_clus.rename({'OCC_CODE':'O*NET-SOC Code'},axis=1, inplace=True)
Un_clus.rename({'OCC_CODE':'O*NET-SOC Code'},axis=1, inplace=True)


#produce indicator lists
Yearlist=[str(i) for i in range(1999,2018)]
#generate rename list to generate wage - year indicators
WageYearlist=[str(i) + ' Av Wage' for i in Yearlist]
EmpYearlist=[str(i) + ' Employment' for i in Yearlist]

#rename the dataframe columns for ease of use
#truncate wagedflist
trunc1=[wagedf_1999, wagedf_2000, wagedf_2001, 
        wagedf_2002, wagedf_2003, wagedf_2004, 
        wagedf_2005, wagedf_2006, wagedf_2007, 
        wagedf_2008, wagedf_2009
        ]

trunc2=[wagedf_2010, wagedf_2011, wagedf_2012,
        wagedf_2013, wagedf_2014, wagedf_2015,
        wagedf_2016, wagedf_2017
        ]

#set renaming dictionaries
rename1={'occ_code':'O*NET-SOC Code', 'tot_emp':'Employment','a_mean':'Av. Wage'}
rename2={'OCC_CODE':'O*NET-SOC Code', 'TOT_EMP':'Employment','A_MEAN':'Av. Wage'}

Renamed1=ColParse(trunc1, rename1)
Renamed2=ColParse(trunc2, rename2)

Renamed1.extend(Renamed2) 
Wagedflist=Renamed1

#generate yearly wageitems
Wage_tseriesdf=dfmerge(Wagedflist,'O*NET-SOC Code',WageYearlist,'Av. Wage')

#add in the cluster identifiers for full time series
Full_Wage_Cluster=df_single_map(Task_clus,Wage_tseriesdf,'O*NET-SOC Code',
                                'TaskWeighted Cluster'
                                )

Full_Wage_Cluster=df_single_map(Un_clus,Wage_tseriesdf,'O*NET-SOC Code',
                                'Cluster'
                                )
#plot the means on the same plot




applylist=list(Full_Wage_Cluster.columns)
del applylist[21]
del applylist[20]
del applylist[0]   



#Clean Wage data
Full_Wage_Cluster.sort_values(['2004 Av Wage'],ascending=False,inplace=True)
Full_Wage_Cluster.reset_index(inplace=True, drop=True)
Full_Wage_Cluster=Full_Wage_Cluster.drop(Full_Wage_Cluster.index[0:5])
#set data to numeric
for i in WageYearlist:
    Full_Wage_Cluster[i]=pd.to_numeric(Full_Wage_Cluster[i])

#turn wages into log wages
Log_Wage_Cluster=dfapplylog(Full_Wage_Cluster,applylist)
    
#find each clusters mean and standard deviation of wages over time
TaskCluster_Meandf=ClusterMean(Full_Wage_Cluster, 'TaskWeighted Cluster',WageYearlist)
UnCluster_Meandf=ClusterMean(Full_Wage_Cluster, 'Cluster', WageYearlist)

TaskCluster_Vardf=ClusterVar(Full_Wage_Cluster, 'TaskWeighted Cluster',WageYearlist)
UnCluster_Vardf=ClusterVar(Full_Wage_Cluster, 'Cluster', WageYearlist)

#set up Cluster Mean Wage and Varience relative to starting values in 1999
TaskCluster_Rel_Meandf=pd.DataFrame(index=TaskCluster_Meandf.index)

#plot the mean and standard deviation of wages over time
Pandaplt(TaskCluster_Meandf,'Year',['Whole Sample','Cluster: 0.0','Cluster: 1.0','Cluster: 4.0','Cluster: 5.0','Cluster: 6.0','Cluster: 7.0','Cluster: 8.0','Cluster: 9.0'])
Pandaplt(TaskCluster_Vardf,'Year',['Whole Sample','Cluster: 0.0','Cluster: 1.0','Cluster: 4.0','Cluster: 5.0','Cluster: 6.0','Cluster: 7.0','Cluster: 8.0','Cluster: 9.0'],mark='.')


#loop through and get relative wage changes and relative varience changes
TaskCluster_Rel_Meandf=Normalisedf(TaskCluster_Meandf)
TaskCluster_Rel_Vardf=Normalisedf(TaskCluster_Vardf)
#re-adjust Year
TaskCluster_Rel_Meandf['Year']=[i for i in range(1999,2018)]
TaskCluster_Rel_Vardf['Year']=[i for i in range(1999,2018)]

Pandaplt(TaskCluster_Rel_Meandf,'Year',['Whole Sample','Cluster: 0.0','Cluster: 1.0','Cluster: 4.0','Cluster: 5.0','Cluster: 6.0','Cluster: 7.0','Cluster: 8.0','Cluster: 9.0'])
Pandaplt(TaskCluster_Rel_Vardf,'Year',['Whole Sample','Cluster: 0.0','Cluster: 1.0','Cluster: 4.0','Cluster: 5.0','Cluster: 6.0','Cluster: 7.0','Cluster: 8.0','Cluster: 9.0'],mark='.')

#look at wage and varience changes relative to the whole sample

TaskCluster_Rel_Meandf['Cluster: 4.0 Diff'] = TaskCluster_Rel_Meandf['Cluster: 4.0']-TaskCluster_Rel_Meandf['Whole Sample']
TaskCluster_Rel_Meandf['Cluster: 0.0 Diff'] = TaskCluster_Rel_Meandf['Cluster: 0.0']-TaskCluster_Rel_Meandf['Whole Sample']
TaskCluster_Rel_Meandf['Cluster: 5.0 Diff'] = TaskCluster_Rel_Meandf['Cluster: 5.0']-TaskCluster_Rel_Meandf['Whole Sample']
TaskCluster_Rel_Meandf['Cluster: 6.0 Diff'] = TaskCluster_Rel_Meandf['Cluster: 6.0']-TaskCluster_Rel_Meandf['Whole Sample']
TaskCluster_Rel_Meandf['Cluster: 7.0 Diff'] = TaskCluster_Rel_Meandf['Cluster: 7.0']-TaskCluster_Rel_Meandf['Whole Sample']
TaskCluster_Rel_Meandf['Cluster: 8.0 Diff'] = TaskCluster_Rel_Meandf['Cluster: 8.0']-TaskCluster_Rel_Meandf['Whole Sample']
TaskCluster_Rel_Meandf['Cluster: 9.0 Diff'] = TaskCluster_Rel_Meandf['Cluster: 9.0']-TaskCluster_Rel_Meandf['Whole Sample']
#do the same for varience
TaskCluster_Rel_Vardf['Cluster: 4.0 Diff'] = TaskCluster_Rel_Vardf['Cluster: 4.0']-TaskCluster_Rel_Vardf['Whole Sample']
TaskCluster_Rel_Vardf['Cluster: 0.0 Diff'] = TaskCluster_Rel_Vardf['Cluster: 0.0']-TaskCluster_Rel_Vardf['Whole Sample']
TaskCluster_Rel_Vardf['Cluster: 5.0 Diff'] = TaskCluster_Rel_Vardf['Cluster: 5.0']-TaskCluster_Rel_Vardf['Whole Sample']
TaskCluster_Rel_Vardf['Cluster: 6.0 Diff'] = TaskCluster_Rel_Vardf['Cluster: 6.0']-TaskCluster_Rel_Vardf['Whole Sample']
TaskCluster_Rel_Vardf['Cluster: 7.0 Diff'] = TaskCluster_Rel_Vardf['Cluster: 7.0']-TaskCluster_Rel_Vardf['Whole Sample']
TaskCluster_Rel_Vardf['Cluster: 8.0 Diff'] = TaskCluster_Rel_Vardf['Cluster: 8.0']-TaskCluster_Rel_Vardf['Whole Sample']
TaskCluster_Rel_Vardf['Cluster: 9.0 Diff'] = TaskCluster_Rel_Vardf['Cluster: 9.0']-TaskCluster_Rel_Vardf['Whole Sample']

#plot relative to whole sample wages and whole sample varience
Pandaplt(TaskCluster_Rel_Meandf,'Year',['Cluster: 0.0 Diff','Cluster: 4.0 Diff','Cluster: 5.0 Diff','Cluster: 6.0 Diff','Cluster: 7.0 Diff','Cluster: 8.0 Diff','Cluster: 9.0 Diff'])
Pandaplt(TaskCluster_Rel_Vardf,'Year',['Cluster: 0.0 Diff','Cluster: 4.0 Diff','Cluster: 5.0 Diff','Cluster: 6.0 Diff','Cluster: 7.0 Diff','Cluster: 8.0 Diff','Cluster: 9.0 Diff'],mark='.')



#Looking at Manual and Abstract Wages Over Time, Mean and Varience
Pandaplt(TaskCluster_Meandf,'Year',['Whole Sample','Cluster: 4.0','Cluster: 9.0'])
Pandaplt(TaskCluster_Vardf,'Year',['Whole Sample','Cluster: 4.0','Cluster: 9.0'])


#Variencedf - without Medicine - Cluster 5.0
Pandaplt(TaskCluster_Vardf,'Year',['Whole Sample','Cluster: 0.0','Cluster: 1.0','Cluster: 4.0','Cluster: 6.0','Cluster: 7.0','Cluster: 8.0','Cluster: 9.0'],mark='.')

Pandaplt(UnCluster_Meandf,'Year',['Cluster: 0','Cluster: 2','Cluster: 3','Cluster: 4','Cluster: 5','Cluster: 6','Cluster: 7','Cluster: 8','Cluster: 9','Whole Sample'])
Pandaplt(UnCluster_Vardf,'Year',['Cluster: 0','Cluster: 5','Cluster: 6','Cluster: 7','Cluster: 9','Whole Sample'],mark='.' )

"""Do the same mean and varience with Log Wages"""

LogTaskCluster_Meandf=ClusterMean(Log_Wage_Cluster, 'TaskWeighted Cluster',WageYearlist)
LogUnCluster_Meandf=ClusterMean(Log_Wage_Cluster, 'Cluster', WageYearlist)

LogTaskCluster_Vardf=ClusterVar(Log_Wage_Cluster, 'TaskWeighted Cluster',WageYearlist)
LogUnCluster_Vardf=ClusterVar(Log_Wage_Cluster, 'Cluster', WageYearlist)

#plot the mean and standard deviation of wages over time
Pandaplt(LogTaskCluster_Meandf,'Year',['Whole Sample','Cluster: 0.0','Cluster: 1.0','Cluster: 4.0','Cluster: 5.0','Cluster: 6.0','Cluster: 7.0','Cluster: 8.0','Cluster: 9.0'])
Pandaplt(LogTaskCluster_Vardf,'Year',['Whole Sample','Cluster: 0.0','Cluster: 1.0','Cluster: 4.0','Cluster: 5.0','Cluster: 6.0','Cluster: 7.0','Cluster: 8.0','Cluster: 9.0'],mark='.')

#Looking at Manual and Abstract Wages Over Time, Mean and Varience
Pandaplt(LogTaskCluster_Meandf,'Year',['Whole Sample','Cluster: 4.0','Cluster: 8.0'])

Pandaplt(LogTaskCluster_Vardf,'Year',['Whole Sample','Cluster: 4.0','Cluster: 8.0'])


#Variencedf - without Medicine - Cluster 5.0
Pandaplt(LogTaskCluster_Vardf,'Year',['Whole Sample','Cluster: 0.0','Cluster: 1.0','Cluster: 4.0','Cluster: 6.0','Cluster: 7.0','Cluster: 8.0','Cluster: 9.0'],mark='.')

Pandaplt(LogUnCluster_Meandf,'Year',['Cluster: 0','Cluster: 2','Cluster: 3','Cluster: 4','Cluster: 5','Cluster: 6','Cluster: 7','Cluster: 8','Cluster: 9','Whole Sample'])
Pandaplt(LogUnCluster_Vardf,'Year',['Cluster: 0','Cluster: 5','Cluster: 6','Cluster: 7','Cluster: 9','Whole Sample'],mark='.' )

#find between Cluster Wage varience over time, for log wages and non log wages
LogTaskBetweenCluster=LogTaskCluster_Meandf.transpose()
TaskBetweenCluster=TaskCluster_Meandf.transpose()

def Betweenclusvarplot(df1):
    df=df1.copy()
    df.drop(['Whole Sample','Year'],inplace=True)
    
    varlist=[]
    for i in list(df.columns):
        var=df[i].std()
        varlist.append(var)
    
    return varlist

    
yearlist=[i for i in range(1999,2018)]
    
    

Logbetweenclusvar = Betweenclusvarplot(LogTaskBetweenCluster)
Betweenclusvar = Betweenclusvarplot(TaskBetweenCluster)

#combine the within cluster and between clusters wage varience over time plots
Task_Within_Betweendf = TaskCluster_Vardf.copy()
Task_Within_Betweendf['Between Cluster Varience'] = Betweenclusvar

Pandaplt(Task_Within_Betweendf,'Year',['Between Cluster Varience','Whole Sample','Cluster: 0.0','Cluster: 4.0','Cluster: 7.0','Cluster: 8.0'],mark='')

Task_Within_Betweendf_Rel_Meandf=Normalisedf(TaskCluster_Meandf)
Task_Within_Betweendf_Rel_Vardf=Normalisedf(Task_Within_Betweendf)
#re-adjust Year
TaskCluster_Rel_Meandf['Year']=[i for i in range(1999,2018)]
Task_Within_Betweendf_Rel_Vardf['Year']=[i for i in range(1999,2018)]

Pandaplt(Task_Within_Betweendf_Rel_Vardf,'Year',['Between Cluster Varience','Cluster: 0.0','Cluster: 4.0','Cluster: 5.0','Cluster: 6.0','Cluster: 7.0','Cluster: 8.0','Cluster: 9.0'])

#now detrend by the whole sample varience over time
Task_Within_Betweendf_Rel_Vardf['Cluster: 4.0 Diff'] = Task_Within_Betweendf_Rel_Vardf['Cluster: 4.0']-Task_Within_Betweendf_Rel_Vardf['Whole Sample']
Task_Within_Betweendf_Rel_Vardf['Cluster: 0.0 Diff'] = Task_Within_Betweendf_Rel_Vardf['Cluster: 0.0']-Task_Within_Betweendf_Rel_Vardf['Whole Sample']
Task_Within_Betweendf_Rel_Vardf['Cluster: 5.0 Diff'] = Task_Within_Betweendf_Rel_Vardf['Cluster: 5.0']-Task_Within_Betweendf_Rel_Vardf['Whole Sample']
Task_Within_Betweendf_Rel_Vardf['Cluster: 6.0 Diff'] = Task_Within_Betweendf_Rel_Vardf['Cluster: 6.0']-Task_Within_Betweendf_Rel_Vardf['Whole Sample']
Task_Within_Betweendf_Rel_Vardf['Cluster: 7.0 Diff'] = Task_Within_Betweendf_Rel_Vardf['Cluster: 7.0']-Task_Within_Betweendf_Rel_Vardf['Whole Sample']
Task_Within_Betweendf_Rel_Vardf['Cluster: 8.0 Diff'] = Task_Within_Betweendf_Rel_Vardf['Cluster: 8.0']-Task_Within_Betweendf_Rel_Vardf['Whole Sample']
Task_Within_Betweendf_Rel_Vardf['Cluster: 9.0 Diff'] = Task_Within_Betweendf_Rel_Vardf['Cluster: 9.0']-Task_Within_Betweendf_Rel_Vardf['Whole Sample']
Task_Within_Betweendf_Rel_Vardf['Between Cluster Diff'] = Task_Within_Betweendf_Rel_Vardf['Between Cluster Varience']-Task_Within_Betweendf_Rel_Vardf['Whole Sample']

#plot relative to whole sample wages and whole sample varience
Pandaplt(Task_Within_Betweendf_Rel_Vardf,'Year',['Between Cluster Diff','Cluster: 0.0 Diff','Cluster: 4.0 Diff','Cluster: 5.0 Diff','Cluster: 6.0 Diff','Cluster: 7.0 Diff','Cluster: 8.0 Diff','Cluster: 9.0 Diff'],mark='.')






LogTask_Within_Betweendf = LogTaskCluster_Vardf.copy()
LogTask_Within_Betweendf['Between Cluster Varience'] = Logbetweenclusvar
Pandaplt(LogTask_Within_Betweendf,'Year',['Between Cluster Varience','Whole Sample','Cluster: 0.0','Cluster: 4.0','Cluster: 7.0','Cluster: 8.0'],mark='')


"""SPARECODE
plt.plot('Year','Cluster: 2.0',data=TaskCluster_Meandf, marker='',color='blue',linewidth=2)
plt.plot('Year','Cluster: 3.0',data=TaskCluster_Meandf, marker='',color='blue',linewidth=2)
plt.legend()
"""


"""COME BACK TO LATER"""
"""
#produce Wage's and Clusters for 2017 only
#produce a copy of the original dataframes
C_wagedf_2017=wagedf_2017.copy()
C_wagedf_2017.rename({'OCC_CODE':'O*NET-SOC Code'},axis=1, inplace=True)
C_Task_clus=Task_clus.copy()
C_Un_clus=Un_clus.copy()

C_Task_clus=df_single_map(C_wagedf_2017,C_Task_clus,'O*NET-SOC Code','A_MEAN')
#clean dataframe up
C_Task_clus=C_Task_clus[C_Task_clus['A_MEAN'] !='*']
C_Task_clus['Mean Wage']=pd.to_numeric(C_Task_clus['A_MEAN'])
ClusterMean()
"""

#produce cluster-level df
#clusters from task weighted
Task_Cluster0=Full_Wage_Cluster[Full_Wage_Cluster['TaskWeighted Cluster']==0]
Task_Cluster1=Full_Wage_Cluster[Full_Wage_Cluster['TaskWeighted Cluster']==1]
Task_Cluster2=Full_Wage_Cluster[Full_Wage_Cluster['TaskWeighted Cluster']==2]
Task_Cluster3=Full_Wage_Cluster[Full_Wage_Cluster['TaskWeighted Cluster']==3]
Task_Cluster4=Full_Wage_Cluster[Full_Wage_Cluster['TaskWeighted Cluster']==4]
Task_Cluster5=Full_Wage_Cluster[Full_Wage_Cluster['TaskWeighted Cluster']==5]
Task_Cluster6=Full_Wage_Cluster[Full_Wage_Cluster['TaskWeighted Cluster']==6]
Task_Cluster7=Full_Wage_Cluster[Full_Wage_Cluster['TaskWeighted Cluster']==7]
Task_Cluster8=Full_Wage_Cluster[Full_Wage_Cluster['TaskWeighted Cluster']==8]
Task_Cluster9=Full_Wage_Cluster[Full_Wage_Cluster['TaskWeighted Cluster']==9]

#clusters from unweighted
Un_Cluster0=Full_Wage_Cluster[Full_Wage_Cluster['Cluster']==0]
Un_Cluster1=Full_Wage_Cluster[Full_Wage_Cluster['Cluster']==1]
Un_Cluster2=Full_Wage_Cluster[Full_Wage_Cluster['Cluster']==2]
Un_Cluster3=Full_Wage_Cluster[Full_Wage_Cluster['Cluster']==3]
Un_Cluster4=Full_Wage_Cluster[Full_Wage_Cluster['Cluster']==4]
Un_Cluster5=Full_Wage_Cluster[Full_Wage_Cluster['Cluster']==5]
Un_Cluster6=Full_Wage_Cluster[Full_Wage_Cluster['Cluster']==6]
Un_Cluster7=Full_Wage_Cluster[Full_Wage_Cluster['Cluster']==7]
Un_Cluster8=Full_Wage_Cluster[Full_Wage_Cluster['Cluster']==8]
Un_Cluster9=Full_Wage_Cluster[Full_Wage_Cluster['Cluster']==9]

    
#define wage vectors   
Wage_1999vec=Full_Wage_Cluster['1999 Av Wage'].values
Wage_2008vec=Full_Wage_Cluster['2008 Av Wage'].values
Wage_2017vec=Full_Wage_Cluster['2017 Av Wage'].values
#define wage vectors for clusters an d#2 years
#0
Wage_Task0_2017=Task_Cluster0['2017 Av Wage'].values
Wage_Task0_2008=Task_Cluster0['2008 Av Wage'].values
Wage_Task0_1999=Task_Cluster0['1999 Av Wage'].values
#1
Wage_Task1_2017=Task_Cluster1['2017 Av Wage'].values
Wage_Task1_2008=Task_Cluster1['2008 Av Wage'].values
Wage_Task1_1999=Task_Cluster1['1999 Av Wage'].values
#3
Wage_Task3_2017=Task_Cluster3['2017 Av Wage'].values
Wage_Task3_2008=Task_Cluster3['2008 Av Wage'].values
Wage_Task3_1999=Task_Cluster3['1999 Av Wage'].values
#4
Wage_Task4_2017=Task_Cluster4['2017 Av Wage'].values
Wage_Task4_2008=Task_Cluster4['2008 Av Wage'].values
Wage_Task4_1999=Task_Cluster4['1999 Av Wage'].values
#5
Wage_Task5_2017=Task_Cluster5['2017 Av Wage'].values
Wage_Task5_2008=Task_Cluster5['2008 Av Wage'].values
Wage_Task5_1999=Task_Cluster5['1999 Av Wage'].values
#6
Wage_Task6_2017=Task_Cluster6['2017 Av Wage'].values
Wage_Task6_2008=Task_Cluster6['2008 Av Wage'].values
Wage_Task6_1999=Task_Cluster6['1999 Av Wage'].values
#7
Wage_Task7_2017=Task_Cluster7['2017 Av Wage'].values
Wage_Task7_2008=Task_Cluster7['2008 Av Wage'].values
Wage_Task7_1999=Task_Cluster7['1999 Av Wage'].values
#8
Wage_Task8_2017=Task_Cluster8['2017 Av Wage'].values
Wage_Task8_2008=Task_Cluster8['2008 Av Wage'].values
Wage_Task8_1999=Task_Cluster8['1999 Av Wage'].values
#9
Wage_Task9_2017=Task_Cluster9['2017 Av Wage'].values
Wage_Task9_2008=Task_Cluster9['2008 Av Wage'].values
Wage_Task9_1999=Task_Cluster9['1999 Av Wage'].values

#create a list calling the dataframes  that have enough occupations
TaskClusterlist_2017=[Wage_Task0_2017,Wage_Task4_2017,Wage_Task5_2017,Wage_Task6_2017,Wage_Task7_2017,Wage_Task8_2017,Wage_Task9_2017]
TaskClusterlist_2008=[Wage_Task0_2008,Wage_Task4_2008,Wage_Task5_2008,Wage_Task6_2008,Wage_Task7_2008,Wage_Task8_2008,Wage_Task9_2008]
TaskClusterlist_1999=[Wage_Task0_1999,Wage_Task4_1999,Wage_Task5_1999,Wage_Task6_1999,Wage_Task7_1999,Wage_Task8_1999,Wage_Task9_1999]
#List of Cluster ID's you have analysed
Clusterlist=['Cluster 0','Cluster 4','Cluster 5','Cluster 6','Cluster 7','Cluster 8','Cluster 9']



#Produce Excel Tables of the Output for 2017-1999,2017-2008,2008-1999
Distests_2017_1999=Disttests(TaskClusterlist_2017,TaskClusterlist_1999,Clusterlist,'TaskClusters, 2017-1999 Disttests')
Distests_2017_2008=Disttests(TaskClusterlist_2017,TaskClusterlist_2008,Clusterlist,'TaskClusters, 2017-2008 Disttests')
Distests_2008_1999=Disttests(TaskClusterlist_2008,TaskClusterlist_1999,Clusterlist,'TaskClusters, 2008-1999 Disttests')

"""Repeat the Analysis Above for Log Wages"""
#produce cluster-level df
#clusters from task weighted
LogTask_Cluster0=Log_Wage_Cluster[Log_Wage_Cluster['TaskWeighted Cluster']==0]
LogTask_Cluster1=Log_Wage_Cluster[Log_Wage_Cluster['TaskWeighted Cluster']==1]
LogTask_Cluster2=Log_Wage_Cluster[Log_Wage_Cluster['TaskWeighted Cluster']==2]
LogTask_Cluster3=Log_Wage_Cluster[Log_Wage_Cluster['TaskWeighted Cluster']==3]
LogTask_Cluster4=Log_Wage_Cluster[Log_Wage_Cluster['TaskWeighted Cluster']==4]
LogTask_Cluster5=Log_Wage_Cluster[Log_Wage_Cluster['TaskWeighted Cluster']==5]
LogTask_Cluster6=Log_Wage_Cluster[Log_Wage_Cluster['TaskWeighted Cluster']==6]
LogTask_Cluster7=Log_Wage_Cluster[Log_Wage_Cluster['TaskWeighted Cluster']==7]
LogTask_Cluster8=Log_Wage_Cluster[Log_Wage_Cluster['TaskWeighted Cluster']==8]
LogTask_Cluster9=Log_Wage_Cluster[Log_Wage_Cluster['TaskWeighted Cluster']==9]

#clusters from unweighted
LogUn_Cluster0=Log_Wage_Cluster[Log_Wage_Cluster['Cluster']==0]
LogUn_Cluster1=Log_Wage_Cluster[Log_Wage_Cluster['Cluster']==1]
LogUn_Cluster2=Log_Wage_Cluster[Log_Wage_Cluster['Cluster']==2]
LogUn_Cluster3=Log_Wage_Cluster[Log_Wage_Cluster['Cluster']==3]
LogUn_Cluster4=Log_Wage_Cluster[Log_Wage_Cluster['Cluster']==4]
LogUn_Cluster5=Log_Wage_Cluster[Log_Wage_Cluster['Cluster']==5]
LogUn_Cluster6=Log_Wage_Cluster[Log_Wage_Cluster['Cluster']==6]
LogUn_Cluster7=Log_Wage_Cluster[Log_Wage_Cluster['Cluster']==7]
LogUn_Cluster8=Log_Wage_Cluster[Log_Wage_Cluster['Cluster']==8]
LogUn_Cluster9=Log_Wage_Cluster[Log_Wage_Cluster['Cluster']==9]

    
#define wage vectors   
LogWage_1999vec=Log_Wage_Cluster['1999 Av Wage'].values
LogWage_2008vec=Log_Wage_Cluster['2008 Av Wage'].values
LogWage_2017vec=Log_Wage_Cluster['2017 Av Wage'].values
#define wage vectors for clusters an d#2 years
#0
LogWage_Task0_2017=LogTask_Cluster0['2017 Av Wage'].values
LogWage_Task0_2008=LogTask_Cluster0['2008 Av Wage'].values
LogWage_Task0_1999=LogTask_Cluster0['1999 Av Wage'].values
#1
LogWage_Task1_2017=LogTask_Cluster1['2017 Av Wage'].values
LogWage_Task1_2008=LogTask_Cluster1['2008 Av Wage'].values
LogWage_Task1_1999=LogTask_Cluster1['1999 Av Wage'].values
#3
LogWage_Task3_2017=LogTask_Cluster3['2017 Av Wage'].values
LogWage_Task3_2008=LogTask_Cluster3['2008 Av Wage'].values
LogWage_Task3_1999=LogTask_Cluster3['1999 Av Wage'].values
#4
LogWage_Task4_2017=LogTask_Cluster4['2017 Av Wage'].values
LogWage_Task4_2008=LogTask_Cluster4['2008 Av Wage'].values
LogWage_Task4_1999=LogTask_Cluster4['1999 Av Wage'].values
#5
LogWage_Task5_2017=LogTask_Cluster5['2017 Av Wage'].values
LogWage_Task5_2008=LogTask_Cluster5['2008 Av Wage'].values
LogWage_Task5_1999=LogTask_Cluster5['1999 Av Wage'].values
#6
LogWage_Task6_2017=LogTask_Cluster6['2017 Av Wage'].values
LogWage_Task6_2008=LogTask_Cluster6['2008 Av Wage'].values
LogWage_Task6_1999=LogTask_Cluster6['1999 Av Wage'].values
#7
LogWage_Task7_2017=LogTask_Cluster7['2017 Av Wage'].values
LogWage_Task7_2008=LogTask_Cluster7['2008 Av Wage'].values
LogWage_Task7_1999=LogTask_Cluster7['1999 Av Wage'].values
#8
LogWage_Task8_2017=LogTask_Cluster8['2017 Av Wage'].values
LogWage_Task8_2008=LogTask_Cluster8['2008 Av Wage'].values
LogWage_Task8_1999=LogTask_Cluster8['1999 Av Wage'].values
#9
LogWage_Task9_2017=LogTask_Cluster9['2017 Av Wage'].values
LogWage_Task9_2008=LogTask_Cluster9['2008 Av Wage'].values
LogWage_Task9_1999=LogTask_Cluster9['1999 Av Wage'].values

#create a list calling the dataframes  that have enough occupations
LogTaskClusterlist_2017=[LogWage_Task0_2017,LogWage_Task4_2017,LogWage_Task5_2017,LogWage_Task6_2017,LogWage_Task7_2017,LogWage_Task8_2017,LogWage_Task9_2017]
LogTaskClusterlist_2008=[LogWage_Task0_2008,LogWage_Task4_2008,LogWage_Task5_2008,LogWage_Task6_2008,LogWage_Task7_2008,LogWage_Task8_2008,LogWage_Task9_2008]
LogTaskClusterlist_1999=[LogWage_Task0_1999,LogWage_Task4_1999,LogWage_Task5_1999,LogWage_Task6_1999,LogWage_Task7_1999,LogWage_Task8_1999,LogWage_Task9_1999]
#List of Cluster ID's you have analysed
Clusterlist=['Cluster 0','Cluster 4','Cluster 5','Cluster 6','Cluster 7','Cluster 8','Cluster 9']



#Produce Excel Tables of the Output for 2017-1999,2017-2008,2008-1999
Distests_2017_1999=Disttests(LogTaskClusterlist_2017,LogTaskClusterlist_1999,Clusterlist,'LogTaskClusters, 2017-1999 Disttests')
Distests_2017_2008=Disttests(LogTaskClusterlist_2017,LogTaskClusterlist_2008,Clusterlist,'LogTaskClusters, 2017-2008 Disttests')
Distests_2008_1999=Disttests(LogTaskClusterlist_2008,LogTaskClusterlist_1999,Clusterlist,'LogTaskClusters, 2008-1999 Disttests') 
    
"""KL DIVERGENCE CALCULATOR SECTION
KEY POINT, TO CALCULATE KL DIVERGENCES YOU NEED TO RUN EACH PIECE OF CODE SEPERATAELY
"""
    
pdf1=PDF_seaborn(Full_Wage_Cluster['1999 Av Wage'])
pdf2=PDF_seaborn(Full_Wage_Cluster['2008 Av Wage'])
test=KL_seaborn(pdf1,pdf2)




"""Subsection: Wage Distribution Graph Production"""

"""Visualization Enviroment - Seaborn"""

#choose style for seaborn plots
sns.set_style("darkgrid")

#generate x/y axis plots:
Pre_pd_task_mean=TaskCluster_Meandf.copy()
Pre_pd_task_mean.index=Yearlist
Pre_pd_task_mean.plot.line()

#produce a series of wage distribution plots 
#Note: run each one to plot seperately, or highlight and run together to plot 
#on the same diagram
sns.distplot(Full_Wage_Cluster['1999 Av Wage'])
sns.distplot(Full_Wage_Cluster['2008 Av Wage'])
sns.distplot(Full_Wage_Cluster['2017 Av Wage'])


#produce a series of per cluster wage distribution plots
#1999 wage distribution of all key clusters
sns.distplot(Task_Cluster0['1999 Av Wage'])
sns.distplot(Task_Cluster4['1999 Av Wage'])
sns.distplot(Task_Cluster5['1999 Av Wage'])
sns.distplot(Task_Cluster6['1999 Av Wage'])
sns.distplot(Task_Cluster7['1999 Av Wage'])
sns.distplot(Task_Cluster8['1999 Av Wage'])
sns.distplot(Task_Cluster5['1999 Av Wage'])


sns.distplot(Full_Wage_Cluster['1999 Av Wage'])

#2017 all clusters
sns.distplot(Task_Cluster0['2017 Av Wage'])
sns.distplot(Task_Cluster4['2017 Av Wage'])
sns.distplot(Task_Cluster5['2017 Av Wage'])
sns.distplot(Task_Cluster6['2017 Av Wage'])
sns.distplot(Task_Cluster7['2017 Av Wage'])
sns.distplot(Task_Cluster8['2017 Av Wage'])
sns.distplot(Task_Cluster5['2017 Av Wage'])

#2017 Engingeeers, Scientists and Manual Workers
sns.distplot(Task_Cluster0['2017 Av Wage'])
sns.distplot(Task_Cluster0['2008 Av Wage'])
sns.distplot(Task_Cluster0['1999 Av Wage'])



sns.distplot(Task_Cluster4['2017 Av Wage'])
sns.distplot(Task_Cluster9['2017 Av Wage'])
sns.distplot(Full_Wage_Cluster['2017 Av Wage'])

#Engineers over time
sns.distplot(Task_Cluster2['1999 Av Wage'])
sns.distplot(Task_Cluster2['2008 Av Wage'])
sns.distplot(Task_Cluster2['2017 Av Wage'])

sns.distplot(Task_Cluster4['2017 Av Wage'])
sns.distplot(Task_Cluster4['2008 Av Wage'])
sns.distplot(Task_Cluster4['1999 Av Wage'])

sns.distplot(Task_Cluster3['1999 Av Wage'])
sns.distplot(Task_Cluster3['2008 Av Wage'])
sns.distplot(Task_Cluster3['2017 Av Wage'])

sns.distplot(Task_Cluster8['1999 Av Wage'])
sns.distplot(Task_Cluster8['2008 Av Wage'])
sns.distplot(Task_Cluster8['2017 Av Wage'])


sns.distplot(Task_Cluster7['1999 Av Wage'])
sns.distplot(Task_Cluster7['2008 Av Wage'])
sns.distplot(Task_Cluster7['2017 Av Wage'])

sns.distplot(Task_Cluster6['1999 Av Wage'])
sns.distplot(Task_Cluster6['2008 Av Wage'])
sns.distplot(Task_Cluster6['2017 Av Wage'])



sns.distplot(Task_Cluster9['1999 Av Wage'])
sns.distplot(Task_Cluster9['2008 Av Wage'])
sns.distplot(Task_Cluster9['2017 Av Wage'])

"""Do the Same Plots for Logs"""

#produce the same plots for log wages
sns.distplot(Log_Wage_Cluster['1999 Av Wage'])
sns.distplot(Log_Wage_Cluster['2008 Av Wage'])
sns.distplot(Log_Wage_Cluster['2017 Av Wage'])

#produce a series of per cluster wage distribution plots
#1999 wage distribution of all key clusters
sns.distplot(LogTask_Cluster0['1999 Av Wage'],kde_kws={'label':'Cluster 0 - 1999'})
sns.distplot(LogTask_Cluster4['1999 Av Wage'],kde_kws={'label':'Cluster 4- 1999'})
sns.distplot(LogTask_Cluster6['1999 Av Wage'],kde_kws={'label':'Cluster 6- 1999'})
sns.distplot(LogTask_Cluster8['1999 Av Wage'],kde_kws={'label':'Cluster 8- 1999'})#
sns.distplot(Log_Wage_Cluster['1999 Av Wage'],kde_kws={'label':'Whole Sample- 1999'})

sns.distplot(LogTask_Cluster0['2017 Av Wage'],kde_kws={'label':'Cluster 0 - 2017'})
sns.distplot(LogTask_Cluster4['2017 Av Wage'],kde_kws={'label':'Cluster 4- 2017'})
sns.distplot(LogTask_Cluster6['2017 Av Wage'],kde_kws={'label':'Cluster 6- 2017'})
sns.distplot(LogTask_Cluster8['2017 Av Wage'],kde_kws={'label':'Cluster 8- 2017'})#
sns.distplot(Log_Wage_Cluster['2017 Av Wage'],kde_kws={'label':'Whole Sample- 2017'})

#look at each clusters change over whole sample
# 0
sns.distplot(LogTask_Cluster0['2017 Av Wage'],kde_kws={'label':'Cluster 0 - 2017'})
sns.distplot(LogTask_Cluster0['1999 Av Wage'],kde_kws={'label':'Cluster 0 - 1999'})
# 4
sns.distplot(LogTask_Cluster4['2017 Av Wage'],kde_kws={'label':'Cluster 4 - 2017'})
sns.distplot(LogTask_Cluster4['1999 Av Wage'],kde_kws={'label':'Cluster 4 - 1999'})
# 5
sns.distplot(LogTask_Cluster5['2017 Av Wage'],kde_kws={'label':'Cluster 5 - 2017'})
sns.distplot(LogTask_Cluster5['1999 Av Wage'],kde_kws={'label':'Cluster 5 - 1999'})
# 6 
sns.distplot(LogTask_Cluster6['2017 Av Wage'],kde_kws={'label':'Cluster 6 - 2017'})
sns.distplot(LogTask_Cluster6['1999 Av Wage'],kde_kws={'label':'Cluster 6 - 1999'})
# 7 
sns.distplot(LogTask_Cluster7['2017 Av Wage'],kde_kws={'label':'Cluster 7 - 2017'})
sns.distplot(LogTask_Cluster7['1999 Av Wage'],kde_kws={'label':'Cluster 7 - 1999'})
# 8
sns.distplot(LogTask_Cluster8['2017 Av Wage'],kde_kws={'label':'Cluster 8 - 2017'})
sns.distplot(LogTask_Cluster8['1999 Av Wage'],kde_kws={'label':'Cluster 8 - 1999'})
# 9
sns.distplot(LogTask_Cluster9['2017 Av Wage'],kde_kws={'label':'Cluster 9 - 2017'})
sns.distplot(LogTask_Cluster9['1999 Av Wage'],kde_kws={'label':'Cluster 9 - 1999'})



#look at each clusters change before and after financial crisis
# 0
sns.distplot(LogTask_Cluster0['2017 Av Wage'],kde_kws={'label':'Cluster 0 - 2017'})
sns.distplot(LogTask_Cluster0['2008 Av Wage'],kde_kws={'label':'Cluster 0 - 2008'})
sns.distplot(LogTask_Cluster0['1999 Av Wage'],kde_kws={'label':'Cluster 0 - 1999'})
# 4
sns.distplot(LogTask_Cluster4['2017 Av Wage'],kde_kws={'label':'Cluster 4 - 2017'})
sns.distplot(LogTask_Cluster4['2008 Av Wage'],kde_kws={'label':'Cluster 4 - 2008'})
sns.distplot(LogTask_Cluster4['1999 Av Wage'],kde_kws={'label':'Cluster 4 - 1999'})
# 5
sns.distplot(LogTask_Cluster5['2017 Av Wage'],kde_kws={'label':'Cluster 5 - 2017'})
sns.distplot(LogTask_Cluster5['2008 Av Wage'],kde_kws={'label':'Cluster 5 - 2008'})
sns.distplot(LogTask_Cluster5['1999 Av Wage'],kde_kws={'label':'Cluster 5 - 1999'})
# 6 
sns.distplot(LogTask_Cluster6['2017 Av Wage'],kde_kws={'label':'Cluster 6 - 2017'})
sns.distplot(LogTask_Cluster6['2008 Av Wage'],kde_kws={'label':'Cluster 6 - 2008'})
sns.distplot(LogTask_Cluster6['1999 Av Wage'],kde_kws={'label':'Cluster 6 - 1999'})
# 7 
sns.distplot(LogTask_Cluster7['2017 Av Wage'],kde_kws={'label':'Cluster 7 - 2017'})
sns.distplot(LogTask_Cluster7['2008 Av Wage'],kde_kws={'label':'Cluster 7 - 2008'})
sns.distplot(LogTask_Cluster7['1999 Av Wage'],kde_kws={'label':'Cluster 7 - 1999'})
# 8
sns.distplot(LogTask_Cluster8['2017 Av Wage'],kde_kws={'label':'Cluster 8 - 2017'})
sns.distplot(LogTask_Cluster8['2008 Av Wage'],kde_kws={'label':'Cluster 8 - 2008'})
sns.distplot(LogTask_Cluster8['1999 Av Wage'],kde_kws={'label':'Cluster 8 - 1999'})
# 9
sns.distplot(LogTask_Cluster9['2017 Av Wage'],kde_kws={'label':'Cluster 9 - 2017'})
sns.distplot(LogTask_Cluster9['1999 Av Wage'],kde_kws={'label':'Cluster 9 - 1999'})






sns.distplot(Log_Wage_Cluster['1999 Av Wage'],kde_kws={'label':'Whole Sample- 1999'})
sns.distplot(LogTask_Cluster7['1999 Av Wage'],kde_kws={'label':'Cluster 7- 1999'})

sns.distplot(Log_Wage_Cluster['2017 Av Wage'],kde_kws={'label':'Whole Sample- 2017'})
sns.distplot(LogTask_Cluster7['2017 Av Wage'],kde_kws={'label':'Cluster 7- 2017'})

sns.distplot(LogTask_Cluster8['2017 Av Wage'],kde_kws={'label':'Cluster 8- 2017'})
sns.distplot(LogTask_Cluster8['1999 Av Wage'],kde_kws={'label':'Cluster 8- 1999'})


sns.distplot(LogFull_Wage_Cluster['1999 Av Wage'])

#2017 all clusters
sns.distplot(LogTask_Cluster0['2017 Av Wage'])
sns.distplot(LogTask_Cluster4['2017 Av Wage'])
sns.distplot(LogTask_Cluster5['2017 Av Wage'])
sns.distplot(LogTask_Cluster6['2017 Av Wage'])
sns.distplot(LogTask_Cluster7['2017 Av Wage'])
sns.distplot(LogTask_Cluster8['2017 Av Wage'])
sns.distplot(LogTask_Cluster5['2017 Av Wage'])
sns.distplot(Log_Wage_Cluster['2017 Av Wage'])


#2017 Engingeeers, Scientists and Manual Workers
sns.distplot(LogTask_Cluster0['2017 Av Wage'])
sns.distplot(LogTask_Cluster0['2008 Av Wage'])
sns.distplot(LogTask_Cluster0['1999 Av Wage'])



sns.distplot(LogTask_Cluster4['2017 Av Wage'])
sns.distplot(LogTask_Cluster9['2017 Av Wage'])
sns.distplot(LogFull_Wage_Cluster['2017 Av Wage'])

#Engineers over time
sns.distplot(LogTask_Cluster2['1999 Av Wage'])
sns.distplot(LogTask_Cluster2['2008 Av Wage'])
sns.distplot(LogTask_Cluster2['2017 Av Wage'])

sns.distplot(LogTask_Cluster4['2017 Av Wage'])
sns.distplot(LogTask_Cluster4['1999 Av Wage'])
sns.distplot(LogTask_Cluster4['2008 Av Wage'])

sns.distplot(LogTask_Cluster3['1999 Av Wage'])
sns.distplot(LogTask_Cluster3['2008 Av Wage'])
sns.distplot(LogTask_Cluster3['2017 Av Wage'])

sns.distplot(LogTask_Cluster5['1999 Av Wage'])
sns.distplot(LogTask_Cluster5['2008 Av Wage'])
sns.distplot(LogTask_Cluster5['2017 Av Wage'])

sns.distplot(LogTask_Cluster6['1999 Av Wage'])
sns.distplot(LogTask_Cluster6['2008 Av Wage'])
sns.distplot(LogTask_Cluster6['2017 Av Wage'])

sns.distplot(LogTask_Cluster7['1999 Av Wage'])
sns.distplot(LogTask_Cluster7['2008 Av Wage'])
sns.distplot(LogTask_Cluster7['2017 Av Wage'])

sns.distplot(LogTask_Cluster9['1999 Av Wage'])
sns.distplot(LogTask_Cluster9['2008 Av Wage'])
sns.distplot(LogTask_Cluster9['2017 Av Wage'])

sns.distplot(LogTask_Cluster8['1999 Av Wage'])
sns.distplot(LogTask_Cluster7['1999 Av Wage'])

sns.distplot(LogTask_Cluster8['2017 Av Wage'])
sns.distplot(LogTask_Cluster7['2017 Av Wage'])


"""Change in Employment Analysis
"""

#generate yearly wageitems
Emp_tseriesdf=dfmerge(Wagedflist,'O*NET-SOC Code',EmpYearlist,'Employment')
#drop rows containing 'all occupations in industry-x'

#find items whose SOC code ends in '0000'
Emp_tseriesdf = Emp_tseriesdf[~Emp_tseriesdf['O*NET-SOC Code'].str.contains('0000')]

 

#map on cluster identifiers


#clean Employment of errors

Emp_tseriesdf=Emp_tseriesdf[Emp_tseriesdf['1999 Employment']!='**']

#make values numeric
for i in EmpYearlist:
    Emp_tseriesdf[i]=pd.to_numeric(Emp_tseriesdf[i])

test1=Full_Emp_Cluster.set_index('O*NET-SOC Code')
# this function normalizes the datafrae of total employment by looking at the share of employment

def TotEmp_EmpShre(Empdf,EmpYearlist):
    #Empdf - dataframe containing employment
    
    df = pd.DataFrame(index=Empdf.index)
    for col in EmpYearlist:
        
        YearEmpshare = [i/sum(list(Empdf[col])) for i in Empdf[col]]
        #test for precision
        assert 1-sum(YearEmpshare) < 0.0001
        df[col]=YearEmpshare

    return df

Emp_sharedf = TotEmp_EmpShre(Emp_tseriesdf,EmpYearlist)
Emp_sharedf['O*NET-SOC Code']=Emp_tseriesdf['O*NET-SOC Code']
Full_Emp_Cluster = df_single_map(Task_clus, Emp_sharedf,'O*NET-SOC Code',
                                'TaskWeighted Cluster'
                                )

Full_Emp_Cluster=df_single_map(Un_clus,Emp_sharedf,'O*NET-SOC Code',
                                'Cluster'
                                )                         

"""Isolate Clusters for Employment Analysis
"""
EmpTask_Cluster0=Full_Emp_Cluster[Full_Emp_Cluster['TaskWeighted Cluster']==0]
EmpTask_Cluster1=Full_Emp_Cluster[Full_Emp_Cluster['TaskWeighted Cluster']==1]
EmpTask_Cluster2=Full_Emp_Cluster[Full_Emp_Cluster['TaskWeighted Cluster']==2]
EmpTask_Cluster3=Full_Emp_Cluster[Full_Emp_Cluster['TaskWeighted Cluster']==3]
EmpTask_Cluster4=Full_Emp_Cluster[Full_Emp_Cluster['TaskWeighted Cluster']==4]
EmpTask_Cluster5=Full_Emp_Cluster[Full_Emp_Cluster['TaskWeighted Cluster']==5]
EmpTask_Cluster6=Full_Emp_Cluster[Full_Emp_Cluster['TaskWeighted Cluster']==6]
EmpTask_Cluster7=Full_Emp_Cluster[Full_Emp_Cluster['TaskWeighted Cluster']==7]
EmpTask_Cluster8=Full_Emp_Cluster[Full_Emp_Cluster['TaskWeighted Cluster']==8]
EmpTask_Cluster9=Full_Emp_Cluster[Full_Emp_Cluster['TaskWeighted Cluster']==9]

#look at clusters total employment shares over time for the clusters
EmpSum_Clusterdf=ClusterSum(Full_Emp_Cluster, 'TaskWeighted Cluster',EmpYearlist)
EmpMean_Clusterdf=ClusterMean(Full_Emp_Cluster, 'TaskWeighted Cluster',EmpYearlist)


Pandaplt(EmpSum_Clusterdf,'Year',['Cluster: 0.0','Cluster: 4.0','Cluster: 5.0',
                                  'Cluster: 6.0','Cluster: 7.0',
                                  'Cluster: 8.0','Cluster: 9.0'],mark='')


Pandaplt(EmpMean_Clusterdf,'Year',['Cluster: 0.0','Cluster: 4.0','Cluster: 5.0',
                                  'Cluster: 6.0','Cluster: 7.0',
                                  'Cluster: 8.0','Cluster: 9.0'],mark='')


Pandaplt(EmpSum_Clusterdf,'Year',['Cluster: 4.0','Cluster: 5.0',
                                  'Cluster: 6.0','Cluster: 7.0',
                                 'Cluster: 8.0'],mark='')


# Normalise by value in 1999 to get index
Norm_Emp_SumCliusterdf=EmpSum_Clusterdf.copy()
Norm_Emp_SumCliusterdf.reset_index(drop=True,inplace=True)



Emp_Norm_Clusters=Normalisedf(EmpSum_Clusterdf)
Emp_Norm_Clusters['Year']=[i for i in range(1999,2018)]
Pandaplt(Emp_Norm_Clusters,'Year',['Cluster: 0.0','Cluster: 4.0','Cluster: 5.0',
                                  'Cluster: 6.0','Cluster: 7.0',
                                  'Cluster: 8.0','Cluster: 9.0'],mark='')



"""Reproduce Autor and Dorn(2013) two graphs of employment and wage changes"""

"""Wages"""
#produce dataframe for normal wages
Percentile_Rankdf=Full_Wage_Cluster.copy()
Percentile_Rankdf.sort_values('1999 Av Wage',inplace=True)
Percentile_Rankdf.reset_index(inplace=True, drop=True)

Percentile_Rankdf['1999 Percentile'] = [i/max(list(Percentile_Rankdf.index)) 
                                        for i in list(Percentile_Rankdf.index)
                                        ]

Percentile_Rankdf['Annual Wage Change']=Percentile_Rankdf['2017 Av Wage']/Percentile_Rankdf['1999 Av Wage']

"""Employment"""
Emp_Percentile_Rankdf=Emp_sharedf.copy()
Emp_Percentile_Rankdf.sort_values('1999 Employment',inplace=True)
Emp_Percentile_Rankdf.reset_index(inplace=True, drop=True)
Emp_Percentile_Rankdf['1999 Percentile'] = Percentile_Rankdf['1999 Percentile']
Emp_Percentile_Rankdf['Annual Employment Change']=Emp_Percentile_Rankdf['2017 Employment']/Emp_Percentile_Rankdf['1999 Employment']

emp_x=list(Emp_Percentile_Rankdf['1999 Percentile'])
emp_y=list(Emp_Percentile_Rankdf['Annual Employment Change'])

sns.regplot(emp_x, emp_y, lowess=True, scatter = False)


sns.distplot(Emp_sharedf['1999 Employment'])

sns.distplot(test1['1999 Employment'])
sns.distplot(test1['2008 Employment'])
sns.distplot(test1['2017 Employment'])




"""Wage Diagram"""
# function to take subdataframe and drop rows where coldrop column value is 
# outside the 5%-95% range
def Droptails(subdf, coldrop, qmin=0.05,qmax=1):
    # subdf - subdf containing the data
    # coldrop - column containing values to examine for dropping
    # qmin - min tail to drop beneath
    # qmax - max tail to drop above
    df=subdf.copy()
    lowerdrop=df[coldrop].quantile(q=qmin)
    upperdrop=df[coldrop].quantile(q=qmax)
    
    df = df[df[coldrop]>lowerdrop]
    
    df = df[df[coldrop]<upperdrop]
    
    return df

Subdf = Percentile_Rankdf[['1999 Percentile','Annual Wage Change']]
TruncRankdf = Droptails(Subdf,'Annual Wage Change',qmax=0.95)

x=list(Percentile_Rankdf['1999 Percentile'])
y=list(Percentile_Rankdf['Annual Wage Change'])

sns.regplot(x, y, lowess=True)

xtest=list(TruncRankdf['1999 Percentile'])
ytest=list(TruncRankdf['Annual Wage Change'])

sns.regplot(xtest, ytest, lowess=True)

Pandaplt(Percentile_Rankdf,'1999 Percentile',['Annual Wage Change'])

# do the same for log wages
LogPercentile_Rankdf=Log_Wage_Cluster.copy()
LogPercentile_Rankdf.sort_values('1999 Av Wage',inplace=True)
LogPercentile_Rankdf.reset_index(inplace=True, drop=True)
LogPercentile_Rankdf['1999 Percentile'] = [i/max(list(LogPercentile_Rankdf.index)) for i in list(LogPercentile_Rankdf.index)]
LogPercentile_Rankdf['Annual Wage Change Ratio']=LogPercentile_Rankdf['2017 Av Wage']/LogPercentile_Rankdf['1999 Av Wage']
LogPercentile_Rankdf['Annual Wage Change Diff']=LogPercentile_Rankdf['2017 Av Wage']-LogPercentile_Rankdf['1999 Av Wage']
LogSubdf = LogPercentile_Rankdf[['1999 Percentile','Annual Wage Change Ratio']]

Pandaplt(LogPercentile_Rankdf,'1999 Percentile',['Annual Wage Change'])

x1=list(LogPercentile_Rankdf['1999 Percentile'])
y1=list(LogPercentile_Rankdf['Annual Wage Change Ratio'])
y2=list(LogPercentile_Rankdf['Annual Wage Change Diff'])

# introduce some floats in our x-values
Percentile_Rankdf['Yearly Wage Change']=(Percentile_Rankdf['Annual Wage Change']/18)*100


x_1 = list(LogPercentile_Rankdf['1999 Percentile'])
y_1 = list(LogPercentile_Rankdf['Annual Wage Change Ratio'])


# lowess will return our "smoothed" data with a y value for at every x-value
lowess = sm.nonparametric.lowess(y, x, frac=.3)

# unpack the lowess smoothed points to their values
lowess_x = list(zip(*lowess))[0]
lowess_y = list(zip(*lowess))[1]
#put in a dataframe from seaborn



# run scipy's interpolation. There is also extrapolation I believe
f = interp1d(lowess_x, lowess_y, bounds_error=False)

xnew = [i/10. for i in range(400)]

# this this generate y values for our xvalues by our interpolator
# it will MISS values outsite of the x window (less than 3, greater than 33)
# There might be a better approach, but you can run a for loop
#and if the value is out of the range, use f(min(lowess_x)) or f(max(lowess_x))
ynew = f(xnew)


plt.plot(x, y, 'o')
plt.plot(lowess_x, lowess_y, '*')
plt.plot(xnew, ynew, '-')
plt.show()

"""Employment Diagram"""


#checking to see log wage distribution
#generate log wage df

"""This function takes a dataframe containing wage time series and produces another dataframe 
containing the log wages
def Logdf(df,indexcol,wagecollist):
pd.DataFrame(index=Full_Wage_Cluster['O*NET-SOC Code'],data=[0]*len(Full_Wage_Cluster.index),columns=['initcol'])
for i in
"""