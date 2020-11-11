#!/usr/bin/env python
# coding: utf-8

# # Author : Shubh Patel
# #Data Science and Business Analysis Internship
# #TASK : 1

# In[1]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


# In[2]:


data = pd.read_csv("http://bit.ly/w-data")


# In[3]:


data.head(5)


# In[4]:


data.plot(x="Hours",y="Scores",style='o')
plt.title('Student Scores')
plt.xlabel("Hours Studied")
plt.ylabel("Student Scored")
plt.show()


# In[5]:


X = data.iloc[:,0].values
Y = data.iloc[:,-1].values
X = X.reshape((X.shape[0],1))
Y = Y.reshape((Y.shape[0],1))


# In[6]:


from sklearn.model_selection import train_test_split
X_train,X_test,Y_tr,Y_ts = train_test_split(X,Y,test_size=0.2,random_state=0)


# In[7]:


from sklearn.linear_model import LinearRegression
reg = LinearRegression()
reg.fit(X_train,Y_tr)


# In[ ]:





# In[16]:


y_pred = reg.predict(X_test)


# In[9]:


SL_predict = reg.predict([[9.25]])
print("Estimated Marks After studying 9.25 hours/day by Scikit-Learn regressor : " +str(SL_predict[0][0]))


# In[10]:


from sklearn import metrics  
print('Mean Absolute Error:', 
      metrics.mean_absolute_error(Y_ts, y_pred)) 


# # Self-Implementation of Simple Linear Regression

# In[17]:


def Simple_linear():
    theta = np.random.rand(1,1)
    b = np.random.rand(1,1)
    alpha = 0.03
    m = X.shape[0]
    for i in range(10000):
        pred = np.matmul(X,theta) + b
        cost = (np.sum((pred - Y)**2))/(2*X.shape[0])
        grad_theta = np.matmul(X.T,(pred-Y))/m
        grad_b = np.sum(pred-Y)/m
        theta = theta - (alpha*grad_theta)
        b = b - (alpha*grad_b)
    line = theta*X + b
    plt.scatter(X,Y)
    plt.plot(X,line)
    plt.title("Custom code")
    plt.show()
    return theta,b


# In[18]:


def find_ans(x,theta=0,b=0,call_func=True):
    if call_func:
        (theta,b) = Simple_linear()
    return (x*theta + b)[0][0]


# In[19]:


self_predict = find_ans(9.25)
print("Estimated Marks After studying 9.25 hours/day by Custom regressor : " +str(self_predict))


# In[14]:


Diff = np.abs((SL_predict - self_predict)[0][0])
print("Total prediction error between Scikit-Learn prediction and Custom prediction : " + str(Diff))
theta,b = Simple_linear()


# In[15]:


line = reg.coef_*X + reg.intercept_
plt.scatter(X,Y)
plt.plot(X,line)
plt.title('Using Sklearn Library')
plt.show()

line = theta*X + b
plt.scatter(X,Y)
plt.plot(X,line)
plt.title('Using Custom Code')
plt.show()


# In[ ]:





# In[ ]:




