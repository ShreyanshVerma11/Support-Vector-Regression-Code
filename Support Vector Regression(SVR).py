#!/usr/bin/env python
# coding: utf-8

# # Importing the libraries

# In[64]:


import numpy as np
import matplotlib.pyplot as plt
import pandas as pd


# # Importing the dataset

# In[65]:


dataset = pd.read_csv('Position_Salaries.csv')
X = dataset.iloc[:, 1:-1].values
y = dataset.iloc[:, -1].values


# In[66]:


print(X)


# In[67]:


print(y)


# In[68]:


y = y.reshape(len(y), 1)


# In[69]:


print(y)


# # Feature Scaling
# 

# In[70]:


from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
X = sc_X.fit_transform(X)


# In[71]:


print(X)


# In[72]:


sc_y = StandardScaler()
y = sc_y.fit_transform(y)


# In[73]:


print(y)


# # Training the SVR model on the whole dataset
# 

# In[74]:


from sklearn.svm import SVR 
regressor = SVR(kernel = 'rbf')
regressor.fit(X, y)


# # Predicting the new result

# In[80]:


sc_y.inverse_transform(regressor.predict(sc_X.transform([[6.5]])))


# # Visualising the SVR results

# In[86]:


plt.scatter(sc_X.inverse_transform(X), sc_y.inverse_transform(y), color = 'red')
plt.plot(sc_X.inverse_transform(X), sc_y.inverse_transform(regressor.predict(X)), color = 'blue')
plt.title("Truth or Bluff (SVR)")
plt.xlabel("Position level")
plt.ylabel("Salary")
plt.show()


# # Visualing the SVR results (for higher resolution and smoother curve)

# In[89]:


X_grid = np.arange(min(sc_X.inverse_transform(X)), max(sc_X.inverse_transform(X)), 0.001)
X_grid = X_grid.reshape((len(X_grid), 1))
plt.scatter(sc_X.inverse_transform(X), sc_y.inverse_transform(y), color = 'red')
plt.plot(X_grid, sc_y.inverse_transform(regressor.predict(sc_X.transform(X_grid))), color = 'blue')
plt.title('Truth or Bluff (Polynomial Regression)')
plt.xlabel('Position level')
plt.ylabel('Salary')
plt.show()


# In[ ]:




