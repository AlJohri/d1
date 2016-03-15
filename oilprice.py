#from pydevsrc import pydevd;pydevd.settrace('192.168.2.8', stdoutToServer=True, stderrToServer=True) #clone and put on python path: https://github.com/tenXer/PyDevSrc

from math import exp

__author__ = 'Liana'

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import statsmodels.api as sm
import scipy.stats as st
import numpy as np
import statsmodels.tsa.arima_process as ap
# from statsmodels.tsa.arima_model import ARIMA
from statsmodels.graphics.api import qqplot

#RK
#import numpy as np
import scipy as sp
from scipy.interpolate import interp1d
import scipy.interpolate as inter
from scipy.signal import argrelextrema
from operator import itemgetter
import operator


def to_integer(dt_time):
#    return 10000*dt_time.year + 1000*dt_time.month + dt_time.day
    return 12*(dt_time.year-1986) + dt_time.month


#def pad(iterable, size, padding=None):
#   return islice(pad_infinite(iterable, padding), size)

# Important: It might be necessary to install xlrd
# pip install xlrd

# Download data from: http://www.eia.gov/dnav/pet/pet_pri_spt_s1_m.htm
# Create an Excel file object
adir='/home/romank/Desktop/DataScience/CrudeOilPricesTimeSeriesAnalysis_Python-master/';
excel = pd.ExcelFile(adir + 'data/PET_PRI_SPT_S1_M.xls' )

# Parse the first sheet
df = excel.parse(excel.sheet_names[1])

# Rename the columns
df = df.rename(columns=dict(zip(df.columns, ['Date','WTI','Brent'])))

# Cut off the first 18 rows because these rows
# contain NaN values for the Brent prices
df = df[18:]

#print df.head()

# Index the data set by date
df.index = df['Date']
x=df['Date']

# Remove the date column after re-indexing
df = df[['WTI','Brent']]

import sys
print 'current trace function', sys.gettrace()

#import pydevd
#pydevd.settrace()
#print df

#===========================
#      VISUALISATION
#===========================
##########################
# RK
###############################################
new_length = 20

y=df['Brent']

#x=to_integer(x.range(start,stop))
#a['x'].apply(lambda x, y: x + y, args=(100,))
datex=x
x=x.apply(lambda x: to_integer(x))

#
#padd the future - to catch the current crisis
#
n_month_padd=14

print(type(x), type(y))

x=np.array(x)
y=np.array(y)

x=np.lib.pad(x, (0,n_month_padd), 'linear_ramp', end_values=(0, x[len(x)-1]+n_month_padd))
y0=np.lib.pad(y.astype(int), (0,n_month_padd), 'edge')
y0=y0.astype(float)
y0[1:len(y)]=y[1:len(y)]
y=y0
print(type(x), type(y))

#fig, ax0 = plt.subplots(figsize=(10,5))
#print(x,y)
#ax0.plot(x, y, c='r', label='Brent')
#ax0.plot(x[1:len(x)], np.diff(x), c='r', label='Brent')

#############################3
# Integrate and find K-curve
#
#new_x = np.linspace(x.min(), x.max(), new_length)
#s1 = inter.InterpolatedUnivariateSpline (x, y)
#s2 = inter.UnivariateSpline (x[::1], y[::1], s=0.1e5)
#dx=x[1]-x[0]
#s2prime_2 = np.gradient(s2(x), dx)
#new_y = sp.interpolate.interp1d(x, s2(x), kind='linear')(new_x)
#y_i=sp.integrate.simps(y, x);
nt=4
y_i=sp.integrate.cumtrapz(y, x);

ny=len(y_i)
y1=y[nt:ny-nt];


nt2=2
nt_k=np.floor(nt/(nt-nt2))
              
#y_i0=y_i[0:ny-nt-nt];
y_i0=y_i[0+nt2:ny-nt-nt+nt2];
y_i1=y_i[nt:ny-nt];
y_i2=y_i[nt+nt:ny];

OIL_PROD_COST=5
#k_i=100*(y_i2-y_i1)/(y_i1-y_i0)
k_i=100*(y_i2-y_i1-nt*OIL_PROD_COST)/(y_i1-y_i0-nt*OIL_PROD_COST)
#k_i=100*(y_i2-y_i1-nt*OIL_PROD_COST)/(nt*y1-nt*OIL_PROD_COST)
#k_i=100*(y_i2-y_i1-OIL_PROD_COST)/nt_k/(y_i1-y_i0-OIL_PROD_COST)

#k_i=k_i*k_i
x_i=x[nt:ny-nt]
datex_i=datex[nt:ny-nt]


################
#
# local min as zero-cross of 1st derivative of K-ratio
#
#yprime_1 = y.diff(x)
#yprime_2 = lambdify(x, yprime, 'numpy')
d_x_i=x_i[1]-x_i[0];
kprime_1=np.diff(k_i) / np.diff(x_i)
kprime_2 = np.gradient(k_i, d_x_i)
# for local minima
#note, these are the indices of x that are local max/min. To get the values, try:
# x[argrelextrema(x, np.greater)[0]]
ind_k_i=argrelextrema(k_i, np.less)
ind_k_i=ind_k_i[0]
ind_k_i=ind_k_i+1
#############################
# Retain TOP local minimums
NUM_LOCAL_MIN=5
#ind_k_i=ind_k_i[0]
top_ind_k_i=ind_k_i[0]

#myList=k_i[ind_k_i]
# list of value-index
#KK1=sorted((e,i) for i,e in enumerate(myList))
Kv=np.array(k_i[ind_k_i]).tolist()
Ki=np.array(ind_k_i).tolist()
KKx=zip(Kv,Ki)
#KKx=[Kv, Ki]
#print(Kv)
#print(Ki)
#print(KKx)

KKxS=sorted((e,i) for e,i in KKx)
#print(KKxS)
KxS=[kkk[1] for kkk in KKxS]
#print(KxS)
top_ind_k_i=KxS[0:NUM_LOCAL_MIN-1]
top_ind_k_i.sort()
#print(top_ind_k_i)

#KK1=sorted(K, key=lambda x: -x[0])
#K.sort(key=itemgetter(1))
#reverse = True
#K.sort()#key=lambda tup: tup[0] )





#print(x_i[top_ind_k_i])
#print(k_i[top_ind_k_i])


#print(top_ind_k_i)
#bestKx.sort()


#print(len(x_i), ind_k_i, K, Kx)
#print(K)

#s2crazy = inter.UnivariateSpline (x[::-1], y[::-1], s=5e8)
#plt.plot (x, y, 'bo', label='Data')
#plt.plot (xx, s1(xx), 'k-', label='Spline, wrong order')
#plt.plot (xx, s1rev(xx), 'k--', label='Spline, correct order')
#plt.plot (xx, s2(xx), 'r-', label='Spline, fit')
# Uncomment to get the poor fit.
#plt.plot (xx, s2crazy(xx), 'r--', label='Spline, fit, s=5e8')
#plt.minorticks_on()
#plt.legend()
#plt.xlabel('x')
#plt.ylabel('y')
#plt.show()




# Use seaborn to control figure aesthetics
sns.set_style("darkgrid")  # try "dark", "whitegrid"

# plot with various axes scales
#plt.figure(1)
fig, ax = plt.subplots(figsize=(15,10))

plt.subplot(221)
plt.plot(datex, df['Brent'], c='r', label='Brent')
plt.title('Crude Oil Prices')
plt.xlabel('Year')
plt.ylabel('Price [USD]')
plt.legend(loc='upper left')

plt.subplot(222)
plt.plot(x, y, c='r', label='Brent (PADDED)')
plt.title('Crude Oil Prices')
plt.xlabel('Months')
plt.ylabel('Price [USD]')
plt.legend(loc='upper left')

#ax.plot(x, s1(x), c='g', label='s1')
#ax.plot(x, s2(x), c='b', label='Smooth spline')
#ax.plot(new_x, new_y, c='g', label='Linear spline')plt.figure(1)

# linear
plt.subplot(223)
#ax.plot(new_x, 30*yprime_2, c='b', label='Derive-1')
plt.plot(x_i, k_i, c='b', label='Crisis as Local-min of K-Ratio: K-Ratio Future profits vs. Past profits')
plt.plot(x_i, kprime_2, c='g', label='Diff-Ratio-Integral')
plt.plot(x_i[ind_k_i], k_i[ind_k_i], '-.' , c='b', label='all local-minimums')
plt.plot(x_i[top_ind_k_i], k_i[top_ind_k_i], '--' , c='r', label='top-five all local-minimums')
plt.title('Crude Oil Prices')
plt.xlabel('Months')
plt.ylabel('Ratio Future profits vs. Past profits [%]')
plt.legend(loc='upper left')


plt.subplot(224)
yB=df['Brent']
plt.plot(datex, df['Brent'], c='r', label='Brent')
plt.plot(datex[top_ind_k_i], yB[top_ind_k_i], c='b', label='Crisis Start Dates')

#plt.plot(x , y , c='r', label='Brent')
#plt.plot(x[top_ind_k_i], y[top_ind_k_i], c='b', label='Brent')

plt.title('Crude Oil Prices')
plt.xlabel('Year')
plt.ylabel('Price [USD]')
plt.legend(loc='upper left')


plt.subplot(223)
#ax.plot(dateKx, 100*K, '-.', c='R', label='K1')

plt.subplot(224)
#ax.plot(datex[nt+ind_k_i-1], y[ind_k_i+nt-1], '-.' , c='b', label='K1')

#ax.plot(df.index, df['WTI'], c='b', label='WTI')

#ax.plot(x, df['Brent'], c='r', label='Brent')
#ax.plot(new_x, new_y, c='b', label='Smooth spline')
#ax.plot(new_x, 40*yprime_2, c='g', label='Deriv Smooth spline')
#ax.plot(df.index, df['Brent'], c='r', label='Brent')



plt.show()


df[-11*4:].to_csv('data/Spot_Prices_2012_2015.csv')

newdf = pd.read_csv('data/Spot_Prices_2012_2015.csv')

dates = pd.Series([pd.to_datetime(date) for date in newdf['Date']])
fig, ax = plt.subplots(figsize=(10,5))
plt.title('Crude Oil Prices')
plt.xlabel('Year')
plt.ylabel('Price [USD]')
ax.plot(dates, newdf['WTI'], c='b', label='WTI')    #np.log()
plt.legend(loc='upper left')
plt.show()


#print newdf.head()



#===========================
#    TIME SERIES ANALYSIS
#===========================

# Building ARIMA model

from statsmodels.tsa.base.datetools import dates_from_range

trainWTI = newdf[:int(0.95*len(newdf))]

# 2012m12 means to start counting months from the 12th month of 2012
# To know the starting month, print trainWTI.head()
dates1 = dates_from_range('2012m1', length=len(trainWTI.WTI))
trainWTI.index = dates1
trainWTI = trainWTI[['WTI']]

print trainWTI.tail()

# Determine whether AR or MA terms are needed to correct any
# autocorrelation that remains in the series.
# Looking at the autocorrelation function (ACF) and partial autocorrelation (PACF) plots of the series,
# it's possible to identify the numbers of AR and/or MA terms that are needed
# In this example, the autocorrelations are significant for a large number of lags,
# but perhaps the autocorrelations at lags 2 and above are merely due to the propagation of the autocorrelation at lag 1.
# This is confirmed by the PACF plot.
# RULES OF THUMB:
# Rule 1: If the PACF of the differenced series displays a sharp cutoff and/or the lag-1 autocorrelation is positive,
# then consider adding an AR term to the model. The lag at which the PACF cuts off is the indicated number of AR terms.
# Rule 2: If the ACF of the differenced series displays a sharp cutoff and/or the lag-1 autocorrelation is negative,
# then consider adding an MA term to the model. The lag at which the ACF cuts off is the indicated number of MA terms.
fig1 = sm.graphics.tsa.plot_acf(trainWTI['WTI'])
ax = fig1.add_subplot(111)
ax.set_xlabel("Lag")
ax.set_ylabel("ACF")
plt.show()

fig2 = sm.graphics.tsa.plot_pacf(trainWTI['WTI'])
ax = fig2.add_subplot(111)
ax.set_xlabel("Lag")
ax.set_ylabel("PACF")
plt.show()


# Parameter freq indicates that monthly statistics is used
arima_mod100 = ARIMA(trainWTI, (2,0,0), freq='M').fit()  # try (1,0,1)
print arima_mod100.summary()

# Check assumptions:
# 1) The residuals are not correlated serially from one observation to the next.
# The Durbin-Watson Statistic is used to test for the presence of serial correlation among the residuals
# The value of the Durbin-Watson statistic ranges from 0 to 4.
# As a general rule of thumb, the residuals are uncorrelated is the Durbin-Watson statistic is approximately 2.
# A value close to 0 indicates strong positive correlation, while a value of 4 indicates strong negative correlation.
print "==================== Durbin-Watson ====================="
print sm.stats.durbin_watson(arima_mod100.resid.values)
print "========================================================"

fig = plt.figure(figsize=(10,5))
ax = fig.add_subplot(111)
ax = arima_mod100.resid.plot(ax=ax)
ax.set_title("Residual series")
plt.show()

resid = arima_mod100.resid

print "============== Residuals normality test ================"
print st.normaltest(resid)
print "========================================================"

fig = plt.figure(figsize=(10,5))
ax = fig.add_subplot(111)
ax.set_title("Residuals test for normality")
fig = qqplot(resid, line='q', ax=ax, fit=True)
plt.show()

fig = plt.figure(figsize=(10,5))
ax = fig.add_subplot(111)
ax = trainWTI.ix['2012':].plot(ax=ax)
fig = arima_mod100.plot_predict('2014m1', '2015m12', dynamic=True, ax=ax, plot_insample=False)
ax.set_title("Prediction of spot prices")
ax.set_xlabel("Dates")
ax.set_ylabel("Price [USD]")
plt.show()