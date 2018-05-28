#!/usr/bin/env python3
# -*- coding: utf-8 -*-

#%%: import libraries
import pandas as pd
import numpy as np
import datetime as dt
import seaborn as sns

#%% set console preferences
pd.set_option('display.width', 200)
sns.set(style='white')

#%% import data and clone working dataframe
df_raw = pd.read_excel('retail-data.xlsx')
df_raw.head()
df = df_raw.copy()

#%% get number of unique countries and their names
df.Country.nunique()
df.Country.unique()

#%% drop duplicates and group by country and customer ID
cc = df[['Country','CustomerID']].drop_duplicates()
cc.groupby(['Country'])['CustomerID']. \
    aggregate('count').reset_index(). \
    sort_values('CustomerID', ascending=False)

#%% isolate to United Kingdom only
df = df.loc[df['Country'] == 'United Kingdom']

#%% remove customers without customer ID
df = df[pd.notnull(df['CustomerID'])]
df.isnull().sum(axis=0)

#%% ensure only positive quantities and prices
df.UnitPrice.min()
df.Quantity.min()
df = df[(df['Quantity']>0)]

#%% check unique value for each column
def unique_counts(df):
   for i in df.columns:
       count = df[i].nunique()
       print(i, ": ", count)
unique_counts(df)

#%% add column for total price
df['TotalPrice'] = df['Quantity'] * df['UnitPrice']

#%% determine first and last order date
df['InvoiceDate'].min()
df['InvoiceDate'].max()

#%% establish day after last purchase as point of calculation for recency
now = dt.datetime(2011,12,10)
df['InvoiceDate'] = pd.to_datetime(df['InvoiceDate'])

#%% create RFM table
rfmTable = df.groupby('CustomerID').agg({
        'InvoiceDate':  lambda x: (now - x.max()).days,      #recency
        'InvoiceNo':    lambda x: len(x),                    #frequency
        'TotalPrice':   lambda x: x.sum()})                  #monetary
rfmTable['InvoiceDate'] = rfmTable['InvoiceDate'].astype(int)

#%% convert invoice date to integer and rename columns for RFM
rfmTable.rename(columns={
        'InvoiceDate':  'Recency', 
        'InvoiceNo':    'Frequency', 
        'TotalPrice':   'Monetary'}, inplace=True)

#%% shift rfmTable data to quantiles for segmentation
quantiles = rfmTable.quantile(q=[0.25,0.5,0.75])
quantiles = quantiles.to_dict()
quantiles

#%% create a segmented RFM table
rfmSegment = rfmTable.copy()

#%% create functions for calculating scores per order
def scoreRecency(x,p,d): # low recency is best and is assigned 1
    if x <= d[p][0.25]:return 1
    elif x <= d[p][0.50]:return 2
    elif x <= d[p][0.75]:return 3
    else:return 4
def scoreFrequency(x,p,d): # high frequency is best and is assigned 1
    if x <= d[p][0.25]:return 4
    elif x <= d[p][0.50]:return 3
    elif x <= d[p][0.75]:return 2
    else:return 1
def scoreMonetary(x,p,d): # high monetary is best and is assigned 1
    if x <= d[p][0.25]:return 4
    elif x <= d[p][0.50]:return 3
    elif x <= d[p][0.75]:return 2
    else:return 1

#%% create new columns for RFM and assign values based on quantile
rfmSegment['R_qt'] = rfmSegment['Recency'].apply(scoreRecency, args=('Recency',quantiles,))
rfmSegment['F_qt'] = rfmSegment['Frequency'].apply(scoreFrequency, args=('Frequency',quantiles,))
rfmSegment['M_qt'] = rfmSegment['Monetary'].apply(scoreMonetary, args=('Monetary',quantiles,))
rfmSegment.head()

#%% calculate total RFM score as string composed of individual RFM quantiles
rfmSegment['RFM'] = rfmSegment.R_qt.map(str) \
                  + rfmSegment.F_qt.map(str) \
                  + rfmSegment.M_qt.map(str)

#%% translate raw RFM values to log values for plotting, common log
rfmSegment = rfmSegment.assign(R_lg = lambda x: np.log10(x.Recency))
rfmSegment = rfmSegment.assign(F_lg = lambda x: np.log10(x.Frequency))
rfmSegment = rfmSegment.assign(M_lg = lambda x: np.log10(x.Monetary))

#%% isolate to high/low value customers and sort by customer monetary value
bestCustomers = rfmSegment[rfmSegment['RFM'].isin(['311', '411'])].sort_values('Monetary', ascending=False)
worstCustomers = rfmSegment[rfmSegment['RFM'] == '444'].sort_values('Monetary', ascending=False)

#%% plot frequency versus monetary of best customers
g = sns.JointGrid(x="F_lg", y="M_lg", data=bestCustomers, size=6)
g = g.plot(sns.regplot, sns.distplot). \
    plot_joint(sns.kdeplot, zorder=0, n_levels=6). \
    set_axis_labels("Frequency (log10)", "Monetary (log10)")

#%% use hue argument to provide a factor variable by RFM score
h = sns.lmplot(x="F_lg", y="M_lg", data=bestCustomers, size=6, hue='RFM'). \
    set_axis_labels("Frequency (log10)", "Monetary (log10)")