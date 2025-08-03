#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt


# #### Load the dataset California Housing

# In[2]:


from sklearn.datasets import fetch_california_housing
housing = fetch_california_housing()


# In[3]:


housing


# In[4]:


print(housing.DESCR)


# In[5]:


print(housing.feature_names)


# In[6]:


print(housing.target)


# In[7]:


print(housing.data)


# #### Prepare the data

# In[8]:


dataset=pd.DataFrame(housing.data, columns=housing.feature_names)


# In[9]:


print(type(dataset))


# In[10]:


dataset.head()


# In[11]:


dataset['Price']=housing.target


# In[12]:


dataset.head()


# In[13]:


dataset.info()


# In[14]:


dataset.describe()


# ##### Check for null values

# In[15]:


dataset.isnull().sum()


# #### EDA: Exploratory Data Analysis
# ###### AveBedrooms and AveRooms = 0.847621
# ###### Longitude and Latitude = -0.924664

# In[16]:


dataset.corr()


# In[17]:


sns.pairplot(dataset)


# #### To detect Outliers in the dataset

# In[18]:


fig, ax=plt.subplots(figsize=(15,15))
sns.boxplot(data=dataset,ax=ax)
plt.savefig('boxplot.png')


# In[19]:


## Split the data into dependent and independent features

x = dataset.iloc[:,:-1]
y = dataset.iloc[:,-1]


# In[20]:


## Split data into train and test set

from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.3,random_state=42)


# In[21]:


x_train


# In[22]:


x_test


# In[23]:


y_train


# In[24]:


y_test


# In[25]:


## Normalization of the given data points

from sklearn.preprocessing import StandardScaler
scaler=StandardScaler()
x_train_norm=scaler.fit_transform(x_train)


# In[26]:


x_train_norm


# In[27]:


fig, ax=plt.subplots(figsize=(15,15))
sns.boxplot(data=x_train_norm,ax=ax)
plt.savefig('boxplot_trainData.png')


# In[28]:


x_test_norm = scaler.transform(x_test)


# In[29]:


fig, ax=plt.subplots(figsize=(15,15))
sns.boxplot(data=x_test_norm,ax=ax)
plt.savefig('boxplot_testData.png')


# ### Model Training

# In[30]:


from sklearn.linear_model import LinearRegression
regression=LinearRegression()
regression.fit(x_train_norm,y_train)


# In[31]:


print(regression.coef_)


# In[32]:


print(regression.intercept_)


# ### Model Pridiction

# In[33]:


reg_pred= regression.predict(x_test_norm)
reg_pred


# In[34]:


## Calculate the errors

residuals= y_test - reg_pred
residuals


# In[35]:


## Distribution plot of the residuals

sns.displot(residuals, kind='kde')


# ### Model Performance

# In[36]:


## Lower error value- MSE and MAE

from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
print(mean_squared_error(y_test,reg_pred))
print(mean_absolute_error(y_test,reg_pred))
print(r2_score(y_test,reg_pred))
print(np.sqrt(mean_squared_error(y_test,reg_pred)))


# In[37]:


score = r2_score(y_test,reg_pred)


# ##### Adjusted R-square

# In[38]:


1-(1-score)*(len(y_test)-1)/(len(y_test)-x_test_norm.shape[1]-1)


# ### Save the model-> Pickle File

# In[39]:


import pickle
pickle.dump(regression,open('model.pkl','wb'))


# ### Load the file and use it for future test data predictions

# In[40]:


model=pickle.load(open('model.pkl','rb'))


# In[48]:


model.predict(scaler.transform(housing.data[0].reshape(1,-1)))


# #### Ragularization: to avoid the overfitting in the modal 

# In[49]:


from sklearn.linear_model import Lasso, Ridge
lasso_regration =Lasso(alpha=1.0)
lasso_regration.fit(x_train_norm, y_train)

ridge_regration = Ridge(alpha=1.0)
ridge_regration.fit(x_train_norm, y_train)


# In[50]:


lasso_pred=lasso_regration.predict(x_test_norm)


# In[51]:


ridge_pred=ridge_regration.predict(x_test_norm)


# In[52]:


print(np.sqrt(mean_squared_error(y_test,lasso_pred)))


# In[53]:


print(np.sqrt(mean_squared_error(y_test,ridge_pred)))


# In[ ]:




