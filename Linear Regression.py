#!/usr/bin/env python
# coding: utf-8

# In[1]:


get_ipython().run_line_magic('matplotlib', 'inline')
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# In[2]:


plt.rcParams['figure.figsize'] = (20.0, 10.0)


# In[3]:


# Reading Data
data = pd.read_csv('headbrain.csv')
print(data.shape)


# In[4]:


print(data)


# In[5]:


data.head()


# In[6]:


# Collecting X and Y
X = data['Head Size(cm^3)'].values
Y = data['Brain Weight(grams)'].values


# In[7]:


mean_x = np.mean(X)
mean_y = np.mean(Y)


# In[8]:


print(mean_x, mean_y)


# In[9]:


# Total no. of values
items = len(X)
items


# ### Solve for b1 and b0
# ### b1 is m where 
# 
# ![image.png](attachment:image.png)
# 
# ### b0 is c 
# ### c = mean_y - (b1 * mean_x)

# In[10]:


# solve for b1 or m
numerator = 0
denominator = 0

for i in range(items):
    numerator += (X[i] - mean_x) * (Y[i] - mean_y)
    denominator += (X[i] - mean_x) ** 2
print(numerator, denominator)


# In[11]:


b1 = numerator / denominator
#b1 = 0.28
b0 = mean_y - (b1 * mean_x)


# In[12]:


# Print Coefficients
print(b1, b0)

# 0.26342933948939945 325.57342104944223 where b1 = numerator / denominator
# 0.27 301.6956962025316 where b1 = 0.27


# In[13]:


# print y predicted values
# y = mx + c

for i in range(items):
    y_prediction = b1 * X[i] + b0
    print(y_prediction)


# In[14]:


# Plotting values and regression line

max_x = np.max(X) + 100
min_x = np.min(X) - 100

# Calculating the line values x and y
x = np.linspace(min_x, max_x, 1000)   # linspace creates sequences of evenly spaced values within a defined interval
y = b0 + b1 * x                       # y_predicted values
print(y)


# In[15]:


# Plotting the line

plt.plot(x, y, color = '#58b970', label = 'Regression line')


# Plotting the scatter points
plt.scatter(X, Y, c = '#ef5423', label = 'Scatter Plot', s = 40)

plt.xlabel('Head Size in cm3')
plt.ylabel('Brain Weight in grams')
plt.legend()
plt.show()


# In[21]:


# Check how good our model is using R2 method or coefficient of determination
# prediction / actual

sum_square_actual = 0
sum_square_predicted = 0
for i in range(items):
    #y_pred_ = y[i]
    y_pred = b0 + b1 * X[i]
    #print(y_pred)
    sum_square_actual += (Y[i] - mean_y) ** 2
    #sum_square_predicted =(y_pred - mean_y) ** 2
    #sum_square_predicted += (Y[i] - y_pred) ** 2
    sum_square_predicted += (y_pred - mean_y) ** 2

r2 = (sum_square_predicted / sum_square_actual)
print(r2)

# 0.26342933948939945 325.57342104944223 where b1 = numerator / denominator
# r2 = 0.6226985655579578

## 0.27 301.6956962025316 where b1 = 0.27
#r2 = 0.6541496414950141


# # using scikit-learn

# In[24]:


from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

X = X.reshape((items, 1))  
# reshape allows us to reshape an array in Python
# In this case, the desired shape is specified as (items, 1), whiere items no. of records in x 
# which means that the resulting array will have 237 rows and column column

regression = LinearRegression()

# Fitting training data
regression = regression.fit(X,Y)

# Y Prediction
y_pred = regression.predict(X)

# Calculating R2 Score
r2_score = regression.score(X,Y)

print(r2)


# In[ ]:




