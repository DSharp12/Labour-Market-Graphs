# -*- coding: utf-8 -*-
"""
Created on Thu Dec 13 10:06:41 2018

@author: 

    Daniel Sharp
"""

"""
This code is a Python-R interface for calculating the probability distribution of
some set of univariate data. It imports a set of distribution-fitting 
functions from R and applies it to a set of values network. It uses the 'goft' 
package from Gonzalez-Estranda and Villasenor (2018) to carry out the data-fitting.

"""

#import modules
import networkx as nx
import numpy as np
import scipy as sp
import pandas as pd
import rpy2.robjects as robjects
from rpy2.robjects import r, pandas2ri
from rpy2.robjects.packages import importr



#import the R packages into Python

#R stores data as vectors - data in the interface needs to be stored as a vector



"""
This function imports a package from R into the python environment.
If you are operating in Conda, the uses the packages which have been
installed into the Conda environment NOT the packages which are installed on
the computer.
Ensure you install package onto the Conda environment using Anaconda Prompt and 
'conda install r-packagename. Then this function imports it into the joint environment.

"""
def import_package(x):
    
    #hello = importr(x)
    
    d = {'print.me': 'print_dot_me', 'print_me': 'print_uscore_me'}
    try:
        name = importr(x, lib_loc = "C:/Users/kiera/Documents/R/win-library/3.5")
    except:
        try:
            name = importr(x, robject_translations = d, lib_loc = "C:/Program Files/R/R-3.5.1/library")
        except:
            print("Error Importing Package")
            
    return name 

"""
KEYNOTE: 
If the package has not been uploaded to Conda beforehand create
it yourself. Locate the R directory the package is installed (normally 'cran').
 1. Open Anaconda Prompt. 
 2. use 'conda skeleton <directoryname> <packagename>'
 (e.g conda skeleton cran goft). This builds a skeleton for the package.
 3.Next write conda build --R=<version> r-<packagename> 
 (e.g conda build --R=3.5.1 r-goft). This builds the package. 
 4.Upload it to your anaconda account. 
 Type 'anaconda upload <path_to_the_package_you_built> -u <your_anaconda_cloud_accountname>. 
 This gives the pathway to the package when the previous code ran. 
 
 This should upload the package to your conda cloud account. Then you have to 
 5. Download the conda package again.
 Type 'conda install -c <your_anaconda_cloud_accountname> r-<packagename>.
"""



  


    
"""
function which turns a numpy array with a single element into an integer
"""
def scal(A):

    b = np.asscalar(A)
    
    return(b)

"""This function tests which distributions a set of data could be drawn from.
It does so uses the 'goft' package in R, which performs goodness of fit tests.

It starts with some more standard distributions and iterates through to some
more exotic distributions. If the p-values are below the level of the
decision rule that you specified it fits the distribution using the fitdistrplus package.
    
So the inputs to the function are a vector of data, A, and a p-value for your
decision rule, p.    
"""    
#'method' can be: 'mle' for 'maximum likelihood estimation', 'mme' for 'moment\
#matching estimation', 'qme' for 'quantile matching estimation', or 'mge' for,\
# 'maximum goodness of fit estimation'    
    
def test_fit(A, p_value, methfunc):
    
    robjects.globalenv['methfunc'] = methfunc
    
    numlist = []
    statementlist = []
    fitlist = []
    
    for i, v in A:
            numlist.append(i)
            if v <= p_value:
               
                b = 'Sufficient evidence to reject the null hypothesis'
                c = ' '
            
            else:
               b = 'Insufficient evidence to reject the null hypothesis'
               
               if i == 0:
                   
                   c = robjects.r('fitdist(A, "norm", method = c(methfunc))')
                   
               elif i == 1:
                   
                   c = robjects.r('fitdist(A, "pareto", method = c(methfunc))')
                   
                   
               elif i == 2: 
                   
                   c = robjects.r('fitdist(A, "lnorm", method = c(methfunc))')
                   
               elif i == 3:
                   
                   c = robjects.r('fitdist(A, "exp", method = c(methfunc))')
                   
               elif i == 4:
                   
                   c = robjects.r('fitdist(A, "gamma", method = c(methfunc))')
                   
               elif i == 5:
                   
                   #LOOK UP FITTING IG TOMORROW
                   c = ' '
                   
               elif i == 6:
                   
                   #LOOK UP FITTING LAPLACE TOMORROW
                   c = ''
                   
               elif i == 7:
                   
                    c = robjects.r('fitdist(A, "cauchy", method = c(methfunc))')
                   
               elif i == 8:
                   
                    c = robjects.r('fitdist(A, "weibull", method = c(methfunc))')
                   
               elif i == 9:
               
                    #Look up gumbell and frechet tomorrow
                    c=''
                   
               elif i == 10:
                   
                   c=''
                   
               else:
                   
                   c = ''
           
            statementlist.append(b)
            fitlist.append(c)
            
    array = list(zip(numlist, statementlist, fitlist))   
    df = pd.DataFrame(array, columns = ['Number', 'Decision', 'Fitted Distribution']) 
    
    return df
            
            
#takes as input a set of tuples of list position and p-values
#checks the p_value, and if the decision rule for rejecting the null
#is reached, then it uses the list position to work out which distribution
#it needs to try to fit to the data, and fits it    

"""
KEYNOTE:
    The pandas to r conversion pandas2ri. Automatically initialises the pandas dataframe as 
    a R data object IN THE PYTHON ENVIRONMENT. It does NOT do so in the R environment. 
    When importing a R function into the Python environment proceed as normal. 
    However when using robjects.r to directly write R code in the Python IDE, then 
    create a copy of the dataframe in the R global environment.
    The line robjects.globalenv['A'] = A does this.
    
"""     
def test_dists(A, p): 
    
    #With the way that goft creates the lists, [[2]] selects the p-value
    
    #creates empty lists to store the 
    numlist = list(range(0,9))
    distlist = []
    rlist = []
    plist = []
    
  
    #feeds the dataset into the R environment
    
    robjects.globalenv['A'] = A
    
    #test whether the data can be represented by a set of
    #distributions in turn
    
    #For each distribution, it carries out the function in R. It then selects
    #the test statistic and p-value from the output given in R, and brings them
    #into the Python environment. Finally, the function appends each of these to
    #a list.
    
    #1. The Normal Distribution
    
    robjects.r('normlist<-normal_test(A)')
    ar = robjects.r('normlist[[1]]')
    ap = robjects.r('normlist[[2]]')
    
    #individual elements of R lists are outputted as a numpy array with one
    #element - this line of code changes it into an integer instead
    ar = scal(ar)
    ap = scal(ap)
    
    distlist.append('Normal Distribution')
    rlist.append(ar)
    plist.append(ap)
    
    #2. The Pareto Distribution
    
    #note the slightly different setup - due to the test outputs being 
    #slightly different too
    
    robjects.r('parlist <-gp_test(A)')
    bp = robjects.r('parlist[[1]]')
 
    
    bp = scal(bp)
    
    distlist.append('Pareto Distribution')
    rlist.append('N/A')
    plist.append(bp)
    
    #3. The Lognormal Distribution
    
    robjects.r('loglist <- lnorm_test(A)')
    cp = robjects.r('loglist[[1]]')

    cp = scal(cp)
    
    distlist.append('Lognormal Distribution')
    rlist.append('N/A')
    plist.append(cp)
    
    #4. The Exponential Distribution
    
    robjects.r('exlist <- exp_test(A)')
    dr = robjects.r('exlist[[1]]')
    dp = robjects.r('exlist[[2]]')
    
    dr = scal(dr)
    dp = scal(dp)
    
    distlist.append('Exponential Distribution')
    rlist.append(dr)
    plist.append(dp)
    
    #5. The Gamma Distribution
    
    robjects.r('gamlist <- gamma_test(A)')
    er = robjects.r('gamlist[[1]]')
    ep = robjects.r('gamlist[[2]]')
    
    er = scal(er)
    ep = scal(ep)
    
    distlist.append('Gamma Distribution')
    rlist.append(er)
    plist.append(ep)
    
    #6. Inverse Gaussian (Wald) Distribution
    
    robjects.r('iglist <- ig_test(A)')
    robjects.r('iglist2<- iglist[[1]]')
    fr = robjects.r('iglist2[[1]]')
    fp = robjects.r('iglist2[[2]]')
    
    fr = scal(fr)
    fp = scal(fp)
    
    distlist.append('Inverse-Gaussian (Wald) Distribution')
    rlist.append(fr)
    plist.append(fp)
    
    #7. Laplace (Double Exponential) Distribution
    
    robjects.r('laplist <- laplace_test(A)')
    gr = robjects.r('laplist[[1]]')
    gp = robjects.r('laplist[[2]]')
    
    gr = scal(gr)
    gp = scal(gp)
    
    distlist.append('Laplace (Double Exponential) Distribution')
    rlist.append(gr)
    plist.append(gp)
    
    #8. Cauchy Distribution
    
    robjects.r('cauchlist <- cauchy_test(A)')
    hr = robjects.r('cauchlist[[1]]')
    hp = robjects.r('cauchlist[[2]]')
    
    hr = scal(hr)
    hp = scal(hp)
    
    distlist.append('Cauchy Distribution')
    rlist.append(hr)
    plist.append(hp)
    
    #9. Weibull Distribution - Non-Extreme Value Version
    
    robjects.r('weilist <- weibull_test(A)')
    ip = robjects.r('weilist[[1]]')
    
    ip = scal(ip)
    
    
    distlist.append ('Weibull Distribution - Non-Extreme-Value Test')
    rlist.append('N/A')
    plist.append(ip)
    
    #10. Weibull Distribution - Extreme Value Test
    
    #robjects.r('weilist2 <- ev_test(A, dist="weibull")')
    #jr = robjects.r('weilist2[[1]]')
    #jp = robjects.r('weilist2[[2]]')
    
    #jr = scal(jr)
    #jp = scal(jp)
    
    #distlist.append('Weibull Distribution - Extreme-Value Test')
    #rlist.append(jr)
    #plist.append(jp)
    
    #11. Gumbel Distribution
    
    robjects.r('gumlist <- ev_test(A, dist="gumbel")')
    kr = robjects.r('gumlist[[1]]')
    kp = robjects.r('gumlist[[2]]')
    
    kr = scal(kr)
    kp = scal(kp)
    
    distlist.append('Gumbel Distribution')
    rlist.append(kr)
    plist.append(kp)
    
    #12. Frecht Distribution
    
    robjects.r('frelist <- ev_test(A, dist="frechet")')
    lr = robjects.r('frelist[[1]]')
    lp = robjects.r('frelist[[2]]')
    
    lr = scal(lr)
    lp = scal(lp)
    
    distlist.append('Frechet Distribution')
    rlist.append(lr)
    plist.append(lp)
    
    fitting = list(zip(numlist, plist))
    
    fitdf = test_fit(fitting, p, "mle")
    
    array = list(zip(numlist, distlist, rlist, plist))
    
    df = pd.DataFrame(array, columns= ['Number', 'Probability Distribution', 'Test Statistic',\
                                       'P-Value'])
    
    df2 = pd.merge(df, fitdf, on = 'Number')
    
    return df2
   
  
    
    #turns the list into a numpy array
    
    #resultsdf = pd.DataFrame(resultslist)
    
    #takes an zips the second and third column of the numpy array into tuples
    
    #p_set = resultsdf.loc[1]
    #names_set = resultsdf.loc[2]
    
    #print(p_set)
    
    #p_zip = list(zip(p_set, names_set))
    
    #takes these tuples and says that if the p_value column of that test is
    #less than the decision rule, you can reject the null hypothesis 
    
    #for i,v in p_zip:
        #if i <= p:
            #print(str(v) + ": can reject the null hypothesis at a " + str(p) + " level.")
            
    #return resultsdf

#def fitlognorm (A):
    
    #lognorm = fitdistrplus.fitdist(A, "lnorm")
    #return lognorm
    
    #statementlist = []
    
 
            
            


"""
Bit where the code is run

"""

#a = robjects.IntVector(list(range(1,100)))



#imports the package and gives it the name we will be using for it

fitdistrplus = import_package('fitdistrplus')

goft = import_package('goft')

stats = import_package('stats')

pandas2ri.activate()

data2 = np.random.normal(10,2,100)

#dataa= pd.DataFrame({'A': [1, 2, 3], 'B': [4, 5, 6], 'C':[7,8,9]},\
                   #index=["one", "two", "three"])
                   
dataa = pd.Series(data2)  
         



#robjects.globalenv['dataa'] = dataa

#robjects.r('A <- names(dataa)[1]')
#robjects.r('B <- names(dataa)[2]')
#robjects.r('linear <- lm(dataa$A~dataa$B)')
#jane = robjects.r('summary(linear)')
#print(jane)

                             

#robjects.r('cauchy <- cauchy_test(df)')

#cauchy2 = robjects.r('summary(cauchy)')


dist_df = test_dists(dataa, 0.05)
dist_df.to_csv("dist.csv")



#test1 = fitlognorm(a)
#print(test1)
