# -*- coding: utf-8 -*-
"""
Created on Fri Mar 25 14:42:54 2022

@author: ASUS
"""

import matplotlib.pyplot as plt
import pandas as pd
from sklearn import datasets
import seaborn as sns
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn import datasets
from factor_analyzer.factor_analyzer import calculate_kmo
from scipy.cluster.hierarchy import dendrogram, linkage
from sklearn.cluster import AgglomerativeClustering
bc=datasets.load_breast_cancer()
bc=pd.DataFrame(data=bc.data,columns= bc.feature_names)
bc.head()
#1.WHat is the shape of this data?
print(bc.shape)
#2. What descriptive statistics you can think of? Find them
print(bc.describe())
#3. DO a pair plot. Which variables show are highly correlated?
print(bc.corr())
#4. What analysis can be done- Cluster Analysis
#Cluster Analysis
bc=np.array(bc)
# converting to array because dataframe doesn't work
Z = linkage(bc, method = "ward")
dendro = dendrogram(Z)
plt.title('Dendrogram')
plt.ylabel('Euclidean distance')
plt.show()
ac = AgglomerativeClustering(n_clusters=3, affinity="euclidean", linkage="ward")
labels = ac.fit_predict(bc)
plt.figure(figsize = (8,5))
plt.scatter(bc[labels == 0,0] , bc[labels == 0,1], c= 'red',label='Cluster 1')
plt.scatter(bc[labels == 1,0] , bc[labels == 1,1], c= 'blue',label='Cluster 2')
plt.scatter(bc[labels == 2,0] , bc[labels == 2,1], c= 'green',label='Cluster 3')
plt.scatter([labels == 3,0] , bc[labels == 3,1], c= 'black',label='Cluster 4')
plt.show()


