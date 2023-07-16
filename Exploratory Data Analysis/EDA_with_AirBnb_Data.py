#!/usr/bin/env python
# coding: utf-8

from __future__ import print_function, division
get_ipython().run_line_magic('matplotlib', 'inline')
import numpy as np
import csv
import pandas as pd
import thinkplot
import thinkstats2
from thinkstats2 import Mean, MeanVar, Var, Std, Cov
import statsmodels.formula.api as smf


dfnyc1 = pd.read_csv('nyc2.csv')

rmtp = dfnyc1

# Room Type
hist = thinkstats2.Hist(dfnyc1.room_type, label='Room Type')
thinkplot.Hist(hist)
thinkplot.Config(xlabel='NYC - Airbnb', ylabel='Count')


# The Entire Home rental type is most popular
# The data appears skewed to the right, this is due to the arbitrary order in which I
# assigned the different room types to numeric data where:
# 1 = entire home/apartment
# 2 = private room within a residence
# 3 = hotel room
# 4 = shared room


#Stats for room type
mean = rmtp.room_type.mean()
var = rmtp.room_type.var()
std = rmtp.room_type.std()

print('The Mean is:', mean, 
      '\nThe Variance is:', var, 
      '\nThe Standard Deviation is:',std)


# Price
hist = thinkstats2.Hist(rmtp.price, label='Price')
thinkplot.Hist(hist)
thinkplot.Config(xlabel='NYC - Airbnb', ylabel='Count')
# Similar to the number of reviews, the pricing is also skewed toward lower 
# pricing with a higher count
hist.Largest()


# Stats for price
mean = rmtp.price.mean()
var = rmtp.price.var()
std = rmtp.price.std()

print('The Mean is:', mean, 
      '\nThe Variance is:', var, 
      '\nThe Standard Deviation is:',std)


# Number of reviews
hist = thinkstats2.Hist(rmtp.num_revs, label='# Reviews')
thinkplot.Hist(hist)
thinkplot.Config(xlabel='NYC - Airbnb', ylabel='Count')
hist.Largest()


# Stats for number of reviews
mean = rmtp.num_revs.mean()
var = rmtp.num_revs.var()
std = rmtp.num_revs.std()

print('The Mean is:', mean, 
      '\nThe Variance is:', var, 
      '\nThe Standard Deviation is:',std)


# Date of last review
hist = thinkstats2.Hist(rmtp.last_review, label='Last Review')
thinkplot.Hist(hist)
thinkplot.Config(xlabel='NYC - Airbnb', ylabel='Count')
# Shows 'entire home' rentals over time, from january 2016-september 2021 in NYC. 
# Count by month



# Reviews per month
hist = thinkstats2.Hist(rmtp.reviews_per_month, label='Reviews Per Month')
thinkplot.Hist(hist)
thinkplot.Config(xlabel='NYC - Airbnb', ylabel='Count')
# I am not discarding outliers for this b/c even though only 3 properties of any
# type have more than 150 reviews in a month average, they're notable for the fact 
# that they are from August and September 2021


# Stats for reviews per month
# Stats for number of reviews
mean = rmtp.reviews_per_month.mean()
var = rmtp.reviews_per_month.var()
std = rmtp.reviews_per_month.std()

print('The Mean is:', mean, 
      '\nThe Variance is:', var, 
      '\nThe Standard Deviation is:',std)



# Using Probability Mass Function (PMF) to compare 2 scenarios for one variable

pmf = thinkstats2.Pmf(rmtp.price, label = 'Price')

thinkplot.Pmf(pmf)
thinkplot.Config(xlabel='Price', ylabel='PMF')



# For entire home rental
rmtp2 = dfnyc1[dfnyc1.room_type == 1]
pmf2 = thinkstats2.Pmf(rmtp2.price, label='Price')

thinkplot.Pmf(pmf2)
thinkplot.Config(xlabel='NYC - Entire Home', ylabel='PMF')


# For private room within a residence
rmtp3 = dfnyc1[dfnyc1.room_type == 2]
pmf3 = thinkstats2.Pmf(rmtp3.price, label='Price')

thinkplot.Pmf(pmf3)
thinkplot.Config(xlabel='NYC - Private Room', ylabel='PMF')


thinkplot.PrePlot(2, cols = 2)
thinkplot.Hist(pmf, align = 'right')
thinkplot.Hist(pmf2, align = 'left')
thinkplot.Config(xlabel = 'Price', 
                 ylabel = 'Probability',
                 axis = [0, 350, 0, .06])


thinkplot.PrePlot(2, cols = 2)
thinkplot.Hist(pmf, align = 'right')
thinkplot.Hist(pmf2, align = 'left')
thinkplot.Config(xlabel = 'Price', 
                 ylabel = 'Probability',
                 axis = [0, 275, 0, .06])


# Cumulative Distribution Function (CDF) for variables
# Number of reviews
cdf = thinkstats2.Cdf(rmtp.num_revs, label = 'Number of Reviews')
thinkplot.Cdf(cdf)
thinkplot.Config(xlabel = 'Number of Reviews', 
                 ylabel = 'CDF', loc = 'lower right')

cdf.Prob(200)

# 98% probability of having 200 or fewer reviews


cdf.Prob(150)

# 95% probability of having 150 or fewer reviews, and so on...



cdf.Prob(50)


cdf.Prob(30)

# The above shows the probability that one will have x-number of reviews or fewer. 


# Price
cdf2 = thinkstats2.Cdf(rmtp.price, label = 'Price')
thinkplot.Cdf(cdf2)
thinkplot.Config(xlabel = 'Price', 
                 ylabel = 'CDF', loc = 'lower right')

cdf2.Prob(200)

# 82% probability of price less than $200


cdf2.Prob(100)

# 47% probability of price less than $100


cdf2.Prob(400)

# The above shows information on price ranges in NYC Airbnbs. There is a 47% probability that
# the price will be below $100/night, while at $400/night there's a 95% probability of finding
# an Airbnb property (of any type).


# This comparison shows that private room pricing is usually lower than pricing for entire residences
entire_cdf = thinkstats2.Cdf(rmtp2.price, label = 'Entire Home')
prv_rm_cdf = thinkstats2.Cdf(rmtp3.price, label = 'Private Room')

thinkplot.PrePlot(2)
thinkplot.Cdfs([entire_cdf, prv_rm_cdf])
thinkplot.Show(xlabel = 'price', ylabel = 'CDF')


# Plotting 1 analytical distribution with analysis of application to my dataset
import scipy.stats

dfnyc1 = pd.read_csv('nyc2.csv')
prices = dfnyc1.price.dropna()


# Normal distribution
mu, var = thinkstats2.TrimmedMeanVar(prices, p=0.01)
print('Mean:', mu, '\nVariance:',var)


# Sigma
sigma = np.sqrt(var)
print('Sigma:', sigma)


xs, ps = thinkstats2.RenderNormalCdf(mu, sigma, low=0, high=12.5)
thinkplot.Plot(xs, ps, label='model', color='0.6')

cdf = thinkstats2.Cdf(prices, label='data')
thinkplot.PrePlot(1)
thinkplot.Cdf(cdf) 
thinkplot.Config(title='Prices',
                 xlabel='Prices',
                 ylabel='CDF',
                 loc = 'lower right')


# Normal Probability Plot below 
mean, var = thinkstats2.TrimmedMeanVar(prices, p = 0.01)
std = np.sqrt(var)

xs = [-5, 5]
fxs, fys = thinkstats2.FitLine(xs, mean, std)
thinkplot.Plot(fxs, fys, linewidth = 4, color = '0.8')

xs, ys = thinkstats2.NormalProbability(prices)
thinkplot.Plot(xs, ys, label = 'All Property Types')
thinkplot.Config(title = 'Normal Probability Plot', 
                 xlabel = 'Standard Deviation from Mean',
                 ylabel = 'Prices')


# Lognormal Model

dfnyc1 = pd.read_csv('nyc2.csv')
price = dfnyc1.price.dropna()

def MakeNormalModel(price):
    ''' Plots a CDF with a Normal model '''
    
    cdf = thinkstats2.Cdf(price, label='Price')

    mean, var = thinkstats2.TrimmedMeanVar(price)
    std = np.sqrt(var)
    print('n, mean, std', len(price), mean, std)

    xmin = mean - 4 * std
    xmax = mean + 4 * std

    xs, ps = thinkstats2.RenderNormalCdf(mean, std, xmin, xmax)
    thinkplot.Plot(xs, ps, label='model', linewidth=4, color='0.8')
    thinkplot.Cdf(cdf)

MakeNormalModel(price)
thinkplot.Config(title='Price', xlabel='Prices',
                 ylabel='CDF', loc='lower right')




# Lognormal price
log_price = np.log10(price)
MakeNormalModel(log_price)
thinkplot.Config(title='Log Scale Prices', xlabel='Prices',
                 ylabel='CDF', loc='upper right')



# Normal probability plot 

def MakeNormalPlot(price):
    # Generates a normal probability plot of birth weights 
    
    mean, var = thinkstats2.TrimmedMeanVar(price, p=0.01)
    std = np.sqrt(var)

    xs = [-5, 5]
    xs, ys = thinkstats2.FitLine(xs, mean, std)
    thinkplot.Plot(xs, ys, color='0.8', label='model')

    xs, ys = thinkstats2.NormalProbability(price)
    thinkplot.Plot(xs, ys, label='prices')

MakeNormalPlot(price)
thinkplot.Config(title='Normal Plot Prices', xlabel='Prices',
                 ylabel='CDF', loc='lower right')



# Lognormal probability plot
MakeNormalPlot(log_price)
thinkplot.Config(title = 'Lognormal Prices',
                 xlabel = 'Prices',
                 ylabel = 'CDF', 
                 loc = 'lower right')



# Scatter plot of price vs. last review date - Shows a significant
# increase in Last Review dates toward late 2021
df = pd.read_csv('nyc2.csv', parse_dates = [4], nrows = None)
df.head()

def SampleRows(df, nrows, replace = False):
    indices = np.random.choice(df.index, nrows, replace = replace)
    sample = df.loc[indices]
    return sample

sample = SampleRows(df, 5000)
last_review, price = sample.last_review, sample.price

thinkplot.Scatter(last_review, price, alpha=1)
thinkplot.Config(xlabel='Last Review',
                 ylabel='Price',
                 legend=False)



# Increase in reviews per month toward late 2021 is noteworthy
sample = SampleRows(df, 5000)
last_review, reviews_per_month = sample.last_review, sample.reviews_per_month

thinkplot.Scatter(last_review, reviews_per_month, alpha=1)
thinkplot.Config(xlabel='Last Review',
                 ylabel='Reviews per Month',
                 legend=False)



# Greater concentration of lower prices with more reviews per month,
# though there are several outliers
sample = SampleRows(df, 5000)
price, reviews_per_month = sample.price, sample.reviews_per_month

thinkplot.Scatter(price, reviews_per_month, alpha=1)
thinkplot.Config(xlabel='Price',
                 ylabel='Reviews per Month',
                 legend=False)



# Price and number of reviews
# In line with previous observations - lower prices garner more reviews
# Opportunity to look into if this is due to broader accessibility to
# lower priced rentals
sample = SampleRows(df, 5000)
price, num_revs = sample.price, sample.num_revs

thinkplot.Scatter(price, num_revs, alpha=1)
thinkplot.Config(xlabel='Price',
                 ylabel='Number of Reviews',
                 legend=False)



# Exploring covariance
cleaned = df.dropna(subset = ['id', 'room_type', 'price', 'num_revs', 'last_review',
                              'reviews_per_month'])

def Cov(xs, ys, meanx=None, meany=None):
    xs = np.asarray(xs)
    ys = np.asarray(ys)

    if meanx is None:
        meanx = np.mean(xs)
    if meany is None:
        meany = np.mean(ys)

    cov = np.dot(xs-meanx, ys-meany) / len(xs)
    return cov




# Price - Room Type
price, room_type = cleaned.price, cleaned.room_type
Cov(price, room_type)



# Price - Reviews per Month
price, reviews_per_month = cleaned.price, cleaned.reviews_per_month
Cov(price, reviews_per_month)



# Price - Number of Reviews
price, num_revs = cleaned.price, cleaned.num_revs
Cov(price, num_revs)



# Pearsons Correlation (PC)
def Corr(xs, ys):
    xs = np.asarray(xs)
    ys = np.asarray(ys)

    meanx, varx = thinkstats2.MeanVar(xs)
    meany, vary = thinkstats2.MeanVar(ys)

    corr = Cov(xs, ys, meanx, meany) / np.sqrt(varx * vary)
    return corr



# PC Price - Number of Reviews
Corr(price, num_revs)



# Correlation coefficient
np.corrcoef(price, num_revs)



# PC Price - Reviews per Month
Corr(price, reviews_per_month)



# Correlation coefficient
np.corrcoef(price, reviews_per_month)



# PC Price - Room Type
Corr(price, room_type)



# Correlation coefficient
np.corrcoef(price, room_type)



# Spearmans Correlation (SC)
def SpearmanCorr(xs, ys):
    xranks = pd.Series(xs).rank()
    yranks = pd.Series(ys).rank()
    return Corr(xranks, yranks)



# SC Price - Number of Reviews
SpearmanCorr(price, num_revs)



# SC Price - Reviews per Month
SpearmanCorr(price, reviews_per_month)



# SC Price - Room Type
SpearmanCorr(price, room_type)



# Hypothesis Testing
class HypothesisTest(object):

    def __init__(self, data):
        self.data = data
        self.MakeModel()
        self.actual = self.TestStatistic(data)

    def PValue(self, iters=1000):
        self.test_stats = [self.TestStatistic(self.RunModel()) 
                           for _ in range(iters)]

        count = sum(1 for x in self.test_stats if x >= self.actual)
        return count / iters

    def TestStatistic(self, data):
        raise UnimplementedMethodException()

    def MakeModel(self):
        pass

    def RunModel(self):
        raise UnimplementedMethodException()

class CoinTest(HypothesisTest):

    def TestStatistic(self, data):
        heads, tails = data
        test_stat = abs(heads - tails)
        return test_stat

    def RunModel(self):
        heads, tails = self.data
        n = heads + tails
        sample = [random.choice('HT') for _ in range(n)]
        hist = thinkstats2.Hist(sample)
        data = hist['H'], hist['T']
        return data

class DiffMeansPermute(thinkstats2.HypothesisTest):

    def TestStatistic(self, data):
        group1, group2 = data
        test_stat = abs(group1.mean() - group2.mean())
        return test_stat

    def MakeModel(self):
        group1, group2 = self.data
        self.n, self.m = len(group1), len(group2)
        self.pool = np.hstack((group1, group2))

    def RunModel(self):
        np.random.shuffle(self.pool)
        data = self.pool[:self.n], self.pool[self.n:]
        return data
  
# Correlation testing
class CorrelationPermute(thinkstats2.HypothesisTest):

   def TestStatistic(self, data):
       xs, ys = data
       test_stat = abs(thinkstats2.Corr(xs, ys))
       return test_stat

   def RunModel(self):
       xs, ys = self.data
       xs = np.random.permutation(xs)
       return xs, ys   


# Reloading CSV 
df = pd.read_csv('nyc5.csv', nrows=None) # this line to convert Last Review
# to a range of numbers that can be used as opposed to the dates

df.head()



cleaned = df.dropna(subset = ['id', 'room_type', 'price', 'num_revs', 'last_review',
                              'reviews_per_month'])



# Price - Last Review  
data = cleaned.price.values, cleaned.last_review.values
ht = CorrelationPermute(data)
pvalue = ht.PValue()
pvalue  

# P Value of 0 - unlikely that these are by chance


# Price - Reviews per Month 
data = cleaned.price.values, cleaned.reviews_per_month.values
ht = CorrelationPermute(data)
pvalue = ht.PValue()
pvalue  

# Relationship possibly by chance



# Price - Number of Reviews  
data = cleaned.price.values, cleaned.num_revs.values
ht = CorrelationPermute(data)
pvalue = ht.PValue()
pvalue  

# Unlikely to be by chance



# Regression Analysis - One Dependent + One Explanatory Var 

# SINGLE REGRESSION

df = pd.read_csv(r'nyc5.csv', nrows = None)  
cleaned = df.dropna(subset = ['id', 'room_type', 'price', 'num_revs', 'last_review',
                              'reviews_per_month'])    
    
price = cleaned.price    
num_revs = cleaned.num_revs    
revs_per_month = cleaned.reviews_per_month    
last_review = cleaned.last_review



def LeastSquares(xs, ys):
    meanx, varx = MeanVar(xs)
    meany = Mean(ys)

    slope = Cov(xs, ys, meanx, meany) / varx
    inter = meany - slope * meanx

    return inter, slope    
    
inter, slope = LeastSquares(price, num_revs)
inter, slope 



inter, slope = LeastSquares(price, revs_per_month)
inter, slope      


inter, slope = LeastSquares(num_revs, revs_per_month)
inter, slope 


inter, slope = LeastSquares(num_revs, price)
inter, slope 


inter, slope = LeastSquares(price, last_review)
inter, slope    

# The dates represented here fall around January 2020


inter, slope = LeastSquares(num_revs, last_review)
inter, slope  

# Date is between November and December of 2019



inter, slope = LeastSquares(revs_per_month, last_review)
inter, slope    

# Date is between December 2019 and January 2020  



def FitLine(xs, inter, slope):
    fit_xs = np.sort(xs)
    fit_ys = inter + slope * fit_xs
    return fit_xs, fit_ys




# Price and Reviews per Month
fit_xs, fit_ys = FitLine(price, inter, slope)

thinkplot.Scatter(price, revs_per_month, color='blue', alpha=0.1, s=10)
thinkplot.Plot(fit_xs, fit_ys, color='white', linewidth=3)
thinkplot.Plot(fit_xs, fit_ys, color='red', linewidth=2)
thinkplot.Config(xlabel="Price",
                 ylabel='Reviews per Month',
                 axis = [0, 1750, 0, 175],
                 legend=False)



# Price and Number of Reviews
fit_xs, fit_ys = FitLine(price, inter, slope)

thinkplot.Scatter(price, num_revs, color='blue', alpha=0.1, s=10)
thinkplot.Plot(fit_xs, fit_ys, color='white', linewidth=3)
thinkplot.Plot(fit_xs, fit_ys, color='red', linewidth=2)
thinkplot.Config(xlabel="Price",
                 ylabel='Number of Reviews',
                 axis = [0, 2000, 0, 300],
                 legend=False)



# Price - Last Review
fit_xs, fit_ys = FitLine(price, inter, slope)

thinkplot.Scatter(price, last_review, color='blue', alpha=0.1, s=10)
thinkplot.Plot(fit_xs, fit_ys, color='white', linewidth=3)
thinkplot.Plot(fit_xs, fit_ys, color='red', linewidth=2)
thinkplot.Config(xlabel="Price",
                 ylabel='Last Review',
                 axis = [0, 2000, 41000, 45000],
                 legend=False)



# MULTIPLE REGRESSION#

#OLS Regression - Number of Reviews / Last Review
df = pd.read_csv(r'nyc5.csv', nrows = None)  
cleaned = df.dropna(subset = ['id', 'room_type', 'price', 'num_revs', 'last_review',
                              'reviews_per_month'])    

formula = 'num_revs ~ last_review'
model = smf.ols(formula, data=df)
results = model.fit()
results.summary()



# OLS Regression - Number of Reviews / Price
formula = 'num_revs ~ price'
model = smf.ols(formula, data=df)
results = model.fit()
results.summary()



# OLS Regression - Last Review / Price
formula = 'last_review ~ price'
model = smf.ols(formula, data=df)
results = model.fit()
results.summary()



# OLS Regression - Last Review / Number of Reviews
formula = 'last_review ~ num_revs'
model = smf.ols(formula, data=df)
results = model.fit()
results.summary()


# ### References
# 
# Downey, A. B. (2011). Think Stats (1st ed.). O’Reilly. 
# 
# Get the Data. Inside Airbnb. (n.d.). http://insideairbnb.com/get-the-data/ 





