
import pandas as pd
import warnings
import datetime as dt

warnings.filterwarnings('ignore')

# The dataset contains all the transactions occurring between 01/12/2010 and 09/12/2011
# for a UK-based and registered online retailer.
df = pd.read_excel("OnlineRetail.xlsx")

print(df.head())
# df0 = df

customer_country = df[['Country', 'CustomerID']].drop_duplicates()
countries = customer_country.groupby(['Country'])['CustomerID'].aggregate('count')\
    .reset_index().sort_values('CustomerID', ascending=False)

print(countries)

# restrict on UK customers only
df1 = df.loc[df['Country'] == 'United Kingdom']
# remove null values
df1 = df1[pd.notnull(df1['CustomerID'])]

# remove values with negative quantity
df1 = df1[(df1['Quantity'] > 0)]
print(df1.shape)
print(df1.info())

# Check unique value for each column.

def unique_counts(df1):

    for i in df1.columns:
       count = df1[i].nunique()
       print(i, ": ", count)

unique_counts(df1)

# Add a column for total price
df1['TotalPrice'] = df1['Quantity'] * df1['UnitPrice']

'''
Find out the first and last order dates in the data
df1['InvoiceDate'].min()
Timestamp(‘2010–12–01 08:26:00’)
df1['InvoiceDate'].max()
Timestamp(‘2011–12–09 12:49:00’)
'''

'''
RECENCY
Since recency is calculated for a point in time, and the last invoice date is 2011–12–09, 
we will use 2011–12–10 as last date to calculate recency.
'''

NOW = dt.datetime(2011, 12, 10)
df1['InvoiceDate'] = pd.to_datetime(df1['InvoiceDate'])

'''
RFM Customer Segmentation - create a RFM table
'''
rfmTable = df1.groupby('CustomerID').agg({'InvoiceDate': lambda x: (NOW - x.max()).days,
                                          'InvoiceNo': lambda x: len(x),
                                          'TotalPrice': lambda x: x.sum()})
rfmTable['InvoiceDate'] = rfmTable['InvoiceDate'].astype(int)
rfmTable.rename(columns={'InvoiceDate': 'recency',
                         'InvoiceNo': 'frequency',
                         'TotalPrice': 'monetary_value'}, inplace=True)

# print("\n\n RFM TABLE IS: \n", rfmTable.head())

# Let’s check the details of the first customer
first_customer = df1[df1['CustomerID'] == 12346]
# print("\n \n first_customer \n" , first_customer)

# Split the metrics
# The easiest way to split metrics into segments is by using quartiles.
# This gives us a starting point for the detailed analysis.
# 4 segments are easy to understand and explain
quantiles = rfmTable.quantile(q=[0.25,0.5,0.75])
quantiles = quantiles.to_dict()

# create a segmented RFM table

segmented_rfm = rfmTable

# The lowest recency, highest frequency and monetary amounts are our best customers.


def RScore(x, p, d):
    if x <= d[p][0.25]:
        return 1
    elif x <= d[p][0.50]:
        return 2
    elif x <= d[p][0.75]:
        return 3
    else:
        return 4


def FMScore(x, p, d):
    if x <= d[p][0.25]:
        return 4
    elif x <= d[p][0.50]:
        return 3
    elif x <= d[p][0.75]:
        return 2
    else:
        return 1


# Add segment numbers to  the  newly created segmented RFM table

segmented_rfm['r_quartile'] = segmented_rfm['recency'].apply(RScore, args=('recency', quantiles,))
segmented_rfm['f_quartile'] = segmented_rfm['frequency'].apply(FMScore, args=('frequency', quantiles,))
segmented_rfm['m_quartile'] = segmented_rfm['monetary_value'].apply(FMScore, args=('monetary_value', quantiles,))


segmented_rfm['RFMScore'] = segmented_rfm.r_quartile.map(str) \
                            + segmented_rfm.f_quartile.map(str) \
                            + segmented_rfm.m_quartile.map(str)

# print(" \n SEGMENTED RFM: \n", segmented_rfm.head())
# ciao

# find out top 10 of our best customers
print(" \n TOP 10 CUSTOMERS FROM SEGMENTED RFM: \n",
      segmented_rfm[segmented_rfm['RFMScore'] == '111'].sort_values('monetary_value', ascending=False).head(10))








