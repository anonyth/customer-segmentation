import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA

#%% set console preferences
pd.set_option('display.width', 400)
pd.set_option('display.max_columns', 10)

#%% import offers data
offers = pd.read_excel('wine-data.xlsx', sheet_name=0)
offers.columns = ['offer_id', 'campaign', 'varietal', 'min_quantity', 'discount', 'origin', 'past_peak']

#%% import transactions data
transactions = pd.read_excel('wine-data.xlsx', sheet_name=1)
transactions.columns = ['customer_name', 'offer_id']
transactions['n'] = 1

#%% merge sets and re-order columns
df = pd.merge(offers, transactions)
df = df[['offer_id', 'campaign', 'customer_name',
         'origin', 'varietal', 'discount', 'min_quantity', 'past_peak', 'n']]

#%% create pivot table to show customer response successes per offer
matrix = df.pivot_table(index=['customer_name'], columns=['offer_id'], values='n')
matrix = matrix.fillna(0).reset_index()
x_columns = matrix.columns[1:]

#%% slice matrix to isolate only the binary indicator column
cluster = KMeans(n_clusters=5)
matrix['cluster'] = cluster.fit_predict(matrix[matrix.columns[2:]])
matrix.cluster.value_counts()

#%% plot counts per cluster
sns.countplot(x='cluster', data=matrix, order=matrix['cluster'].value_counts().index)
plt.show()

#%% transform to 2 dimensional dataset via principal component analysis
pca = PCA(n_components=2)
matrix['x'] = pca.fit_transform(matrix[x_columns])[:,0]
matrix['y'] = pca.fit_transform(matrix[x_columns])[:,1]
matrix = matrix.reset_index()

#%% create new df with name, cluster membership, and coordinates
customer_clusters = matrix[['customer_name', 'cluster', 'x', 'y']]
customer_clusters.head()

#%% merge dataframes
df = pd.merge(transactions, customer_clusters)
df = pd.merge(offers, df)

#%% create scatterplot
sns.lmplot('x', 'y', data=df, hue='cluster', fit_reg=False, palette='muted', legend=True, size=4)
plt.show()

#%% explore particular cluster
df['is_4'] = df.cluster==4
df.groupby("is_4").varietal.value_counts()
df.groupby("is_4")[['min_quantity', 'discount']].mean()