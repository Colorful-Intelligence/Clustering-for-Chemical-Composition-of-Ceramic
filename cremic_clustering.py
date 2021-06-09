#%% Import Libraries

import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
plt.style.use('seaborn-whitegrid')

from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler , normalize

import warnings
warnings.filterwarnings("ignore")

#%% Read the dataset

data = pd.read_csv("chemical _composion_of_ceramic.csv")

#%% Exploratory Data Analysis (EDA)

data.shape # (88, 19)

data.info()

data.describe()
data.drop(["Ceramic Name"],axis = 1,inplace = True)

data["Part"] = [1 if i.strip() == "Body" else 0 for i in data.Part]

#%% Standardize Data

scaler = StandardScaler()
scaler_df = scaler.fit_transform(data)

#%% Normalizing Data

normalized_df = normalize(scaler_df)

#%% Principal Compenent Analysis (PCA)

from sklearn.decomposition import PCA

pca = PCA(n_components = 2) 
X_principal = pca.fit_transform(normalized_df) 
X_principal = pd.DataFrame(X_principal) 
X_principal.columns = ['P1', 'P2']

#%% K-Means Clustering Algorithm

wcss = []

for i in range(1,11):
    kmeans = KMeans(n_clusters = i, init = 'k-means++', max_iter=200, n_init=10,random_state=0)
    kmeans.fit(X_principal)
    wcss.append(kmeans.inertia_)

plt.plot(range(1,11),wcss)
plt.title("Elbow Method")
plt.xlabel("Number of Cluster")
plt.ylabel("wcss")
plt.show()

#%%
kmeans = KMeans(n_clusters=3)
kmeans.fit(X_principal)

plt.scatter(X_principal['P1'], X_principal['P2'],  
           c = KMeans(n_clusters = 3).fit_predict(X_principal), cmap =plt.cm.winter) 

plt.show() 

#plotting the centroid of the clusters
plt.scatter(kmeans.cluster_centers_[:,0],kmeans.cluster_centers_[:,1], s = 100, c= 'red', label = 'centroids')
plt.title('K-means For Chemical Composion of Ceramics')
plt.legend()
plt.show()







