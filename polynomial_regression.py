#!/usr/bin/env python
# coding: utf-8

# ## importing the liabraries

# In[2]:


import numpy as np
import matplotlib.pyplot as plt
import pandas as pd


# ## Importing the dataset

# In[3]:


dataset = pd.read_csv('Position_Salaries.csv')
x = dataset.iloc[:, 1:-1].values
y = dataset.iloc[:, -1].values


# In[4]:


print(x)


# In[5]:


print(y)


# ## Training the linear Regression model on the whole dataset

# In[6]:


from sklearn.linear_model import LinearRegression
lin_reg = LinearRegression()
lin_reg.fit(x,y)


# ## Training the polynomial linear Regression model on the whole dataset

# In[7]:


from sklearn.preprocessing import PolynomialFeatures
poly_reg = PolynomialFeatures(degree = 4)
x_poly = poly_reg.fit_transform(x)
lin_reg_2 = LinearRegression()
lin_reg_2.fit(x_poly, y)


# ## Visualising the Linear Regression results

# In[8]:


plt.scatter(x, y, color = 'red')
plt.plot(x, lin_reg.predict(x), color = 'blue')
plt.title('Truth or Bluff (Linear Regression)')
plt.xlabel('position Level')
plt.ylabel('salary')
plt.show()


# note = predicted salary are very far from the original salary(Blue slope), only one points are very near to the original salary except all are far from original salary, clearly not adapted

# ## Visualising the Polynomial Regression results

# In[9]:


plt.scatter(x, y, color = 'red')
plt.plot(x, lin_reg_2.predict(poly_reg.fit_transform(x)), color = 'blue')
plt.title('Truth or Bluff (polynomial Regression)')
plt.xlabel('position Level')
plt.ylabel('salary')
plt.show()


# note: predicted salary are perfectly fitting to the original salary(Blue curve) of polynomial reg line, here perfectly prediction of salary position level 6 and 7..

# ## Visualising the polynomial Regression results (for higher resolution and smoother curve)

# In[10]:


x_grid = np.arange(min(x), max(x), 0.1)
x_grid = x_grid.reshape((len(x_grid),1))
plt.scatter(x, y, color = 'red')
plt.plot(x, lin_reg_2.predict(poly_reg.fit_transform(x)), color = 'blue')
plt.title('Truth or Bluff (polynomial Regression)')
plt.xlabel('position Level')
plt.ylabel('salary')
plt.show()


# note: we have very well trained and therefore very wellfitted but over fitted model for this dataset

# ## Predicting a new result with Linear Regression

# In[11]:


lin_reg.predict([[6.5]])


# ## Predicting a new result with polynomial Regression

# In[12]:


lin_reg_2.predict(poly_reg.fit_transform([[6.5]])) 


# In[ ]:




