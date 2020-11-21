#!/usr/bin/env python
# coding: utf-8

# # Shubh Patel
# 
# TASK 2
# 

# In[1]:


import pandas as pd
from sklearn import datasets


# In[2]:


import matplotlib.pyplot as plt


# # Importing Data

# In[3]:


data = datasets.load_iris()
data = pd.DataFrame(data.data,columns = data.feature_names)


# In[4]:


data.head(5)
x = data.iloc[:,[0,1,2,3]].values


# In[5]:


from sklearn.cluster import KMeans
check = []


# # Cheking Appropriate Number Of Groups For That Using Elbow Method

# In[6]:


for i in range(1,11):
    kmeans = KMeans(n_clusters=i,max_iter=300,random_state=0)
    kmeans.fit(data.iloc[:,[0,1,2,3]].values)
    check.append(kmeans.inertia_)


# In[7]:


plt.plot(range(1,11),check)
plt.show()


# # By Analysing Above Graph We Can Say There Are 3 Groups By Looking At Elbow

# In[8]:


kmeans = KMeans(n_clusters = 3, 
                max_iter = 10000, n_init = 10, random_state = 0)
y_kmeans = kmeans.fit_predict(data.iloc[:,[0,1,2,3]].values)


# # Plotting Graph With 3 Clusters And Their Clusters

# In[9]:


plt.scatter(x[y_kmeans == 0, 0], x[y_kmeans == 0, 1], 
            s = 100, c = 'red', label = 'Iris-setosa')
plt.scatter(x[y_kmeans == 1, 0], x[y_kmeans == 1, 1], 
            s = 100, c = 'blue', label = 'Iris-versicolour')
plt.scatter(x[y_kmeans == 2, 0], x[y_kmeans == 2, 1],
            s = 100, c = 'green', label = 'Iris-virginica')
plt.scatter(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:,1], 
            s = 100, c = 'yellow', label = 'Centroids')

plt.legend()


# KMeans Algorithm 
# 
# 1. First randomaly initialize and choose centroids as found by elbow method
# 2. then, Assign each data point to their nearest centroids and can find nearest centroids using euclidean distance
# 3. then, take avarage of all data points of all individual centroids one by one and then assign move individual centroids to their avarage
# 4. then, keep doing above and steps untill a stable point is found

# In[ ]:




