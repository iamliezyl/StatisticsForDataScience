#!/usr/bin/env python
# coding: utf-8

# In[1]:


import sklearn
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt


# ![image.png](attachment:image.png)

# In[2]:


# Seed is a repeatable one when generating sample data
np.random.seed(50)

# this will generate random numbers between 0 and 1
X = 2 * np.random.rand(100, 1)

y = 4 + 3 * X + np.random.randn(100,1)  #add noise in the data  (generate random numbers with normal distribution)


# In[3]:


plt.plot(X,y,'bo')
plt.xlabel("X")
plt.ylabel("Y")
plt.axis([0,2,0,15])
plt.show()


# In[4]:


X.shape


# In[11]:


# To get the intercept, we need to have more than 1 feature, so we will add 1 to each of the instance
X_b = np.c_[np.ones((100,1)), X]
X_b.shape


# In[12]:


X_b


# In[29]:


# solve for theta best

#linalg - linear algebra; inv - inverse
theta_best = np.linalg.inv(X_b.T.dot(X_b)).dot(X_b.T).dot(y)


# In[30]:


# result gives us intercept and coefficient
theta_best


# In[36]:


# create random data points

X_new = np.array([[0],[3]])
X_new_b = np.c_[np.ones((2,1)),X_new]


# In[37]:


X_new_b


# In[38]:


#solve for y_predict

y_predict = X_new_b.dot(theta_best)
y_predict


# In[47]:


# plot the visualization

plt.plot(X_new, y_predict, "r-", label = "regression line")
plt.plot(X,y,'bo')
plt.xlabel("X")
plt.ylabel("y")
plt.axis([0,2,0,15])
plt.legend()
plt.show()


# In[ ]:




