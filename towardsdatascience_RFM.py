
import pandas as pd
import warnings
import datetime as dt
import numpy as np


import matplotlib.pyplot as plt
import seaborn as sns; sns.set()  # for plot styling

warnings.filterwarnings('ignore')

# The dataset contains all the transactions occurring between 01/12/2010 and 09/12/2011
# for a UK-based and registered online retailer.
df = pd.read_excel("OnlineRetail.xlsx")
#df[list('CustomerID')] = df[list('CustomerID')].astype(int)

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


# find out top 10 of our best customers
print(" \n TOP 10 CUSTOMERS FROM SEGMENTED RFM: \n",
      segmented_rfm[segmented_rfm['RFMScore'] == '111'].sort_values('monetary_value', ascending=False).head(10))



cluster_df = segmented_rfm[['r_quartile', 'f_quartile', 'm_quartile']]

print(" \n ******** \n CLUSTER SEGMENTED RFM: \n ********** \n ",
      cluster_df.head(10))

from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_samples, silhouette_score

# numero di cluster ipotizzabili
scores = pd.DataFrame()
scores['n_clusters'] = [3, 4, 5, 6, 7]

for k in scores['n_clusters']:
    # Create a kmeans model on our data, using k clusters.  random_state helps ensure that the algorithm returns the same results each time.
    kmeans_model = KMeans(n_clusters=k, random_state=10).fit(cluster_df.iloc[:, :])

    # These are our fitted labels for clusters -- the first cluster has label 0, and the second has label 1.
    labels = kmeans_model.labels_

    # Sum of distances of samples to their closest cluster center
    inertia= kmeans_model.inertia_
    print("n_clusters:", k, " cost:", inertia / 10000000000)
    scores.loc[scores['n_clusters'] == k, 'score'] = inertia / 10000000000


plt.plot(scores['n_clusters'], scores['score'])

#plt.plot(scores) # plotting by columns
plt.title('The Elbow Method')
plt.xlabel('Number of clusters K')
plt.ylabel('Average Within-Cluster distance to Centroid (WCSS)')
plt.show()




# fissiamo N CLUSTERS:
N_CL = 6
kmeans = KMeans(n_clusters=N_CL, random_state=10)
kmeans.fit(cluster_df)
y_kmeans = kmeans.predict(cluster_df)

cluster_df['kmeans_cluster'] = y_kmeans

print(" \n ******** \n CLUSTER SEGMENTED RFM: \n ********** \n ",
      cluster_df.head(10))


print(cluster_df.groupby(['kmeans_cluster']).size())

# cluster centers
centers = np.array(kmeans_model.cluster_centers_)
print(" Centers:  ", centers)

