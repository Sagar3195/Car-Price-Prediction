#!/usr/bin/env python
# coding: utf-8

# In[ ]:





# In[ ]:





# In[1]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')
get_ipython().run_line_magic('matplotlib', 'inline')


# In[4]:


#loading dataset
data = pd.read_csv("car_data.csv")


# In[3]:


data.head(20)


# In[5]:


data.shape


# In[6]:


#target variable is selling price.


# In[9]:


data.dtypes


# In[10]:


print(data['Seller_Type'].unique())
print(data['Transmission'].unique())
print(data['Fuel_Type'].unique())
print(data['Owner'].unique())


# In[11]:


#also we can see the unique value of features using for loop
unique_variable = ['Seller_Type', 'Transmission', 'Fuel_Type', 'Owner']
for feature in data[unique_variable]:
    print(data[feature].unique())


# In[12]:


#now checking missing values in dataset
data.isnull().sum()


# In[13]:


#we can see that there is no missing values


# In[14]:


data.describe()


# In[ ]:





# In[15]:


data.columns


# In[16]:


#we remove one feature car name which is not required for our model.s
final_data = data[['Year', 'Selling_Price', 'Present_Price', 'Kms_Driven',
       'Fuel_Type', 'Seller_Type', 'Transmission', 'Owner']]


# In[17]:


final_data.head()


# In[ ]:





# In[18]:


final_data['Current_Year'] = 2020


# In[19]:


final_data.head()


# In[20]:


#Here we create new feature to know the how old car is.
final_data['No_Of_Year'] = final_data['Current_Year'] - final_data['Year']


# In[21]:


final_data.head()


# In[ ]:





# In[22]:


final_data.drop(['Year', 'Current_Year'], axis = 1, inplace = True)


# In[23]:


final_data.head()


# In[24]:


final_data.shape


# In[25]:


final_data.head()


# In[26]:


#Now we convert all categorical variables into numerical variables
final_data = pd.get_dummies(final_data, drop_first= True)


# In[27]:


final_data.head()


# In[28]:


final_data.shape


# In[29]:


final_data.dtypes


# In[30]:


corrmat = data.corr()
top_corr_features = corrmat.index


# In[31]:


plt.figure(figsize = (12,10))
sns.heatmap(data[top_corr_features].corr(), annot = True, cmap = "RdYlGn")


# #### We can see that there is correlation between Present_Price and Selling Price.

# In[32]:


#Now splitting dataset into independent variable and dependent variable
X = final_data.iloc[:,1 :]
y = final_data.iloc[:,0]


# In[33]:


X.head()


# In[34]:


y.head()


# In[35]:


X.shape, y.shape


# In[36]:


#now lets see the feature importance
from sklearn.ensemble import ExtraTreesRegressor
model = ExtraTreesRegressor()


# In[37]:


model.fit(X,y)


# In[38]:


print(model.feature_importances_)


# In[39]:


#lets plot the feature importance for better visualization
feat_importance = pd.Series(model.feature_importances_, index = X.columns)
feat_importance


# In[40]:


feat_importance.nlargest(5).plot(kind = 'barh')


# In[41]:


#we can see that present_price have more impotrance feature, next feature is seller type indvidual.


# In[ ]:





# In[42]:


#Now we splitting dataset into training data and testing data
from sklearn.model_selection import train_test_split
x_train, x_test,y_train, y_test = train_test_split(X,y, test_size = 0.2)


# In[43]:


x_train.shape, x_test.shape


# In[44]:


#Here we use randomforest regression model
from sklearn.ensemble import RandomForestRegressor


# In[45]:


#now create instance of model
rf_regressor = RandomForestRegressor()


# In[46]:


#here we do hyperparameter tunning using randomized search cv
from sklearn.model_selection import RandomizedSearchCV


# In[47]:


#No of tree for in random forest
n_estimators = [int(x) for x in np.linspace(start = 100, stop = 1200, num = 12)]
#No of features consider at every split
max_features = ['auto', 'sqrt']
#maximum number of level in tree
max_depth = [int(x) for x in np.linspace(start = 5, stop = 30, num = 6)]
#minimum number of samples required to split to nodes
min_samples_split = [2,5,10,15] 
#minimum number of samples required at each leaf node.
min_samples_leaf = [1,2,45,10] 


# In[48]:


#now create random grid
random_grid = {'n_estimators': n_estimators,
               'max_features': max_features,
               'max_depth': max_depth,
               'min_samples_split': min_samples_split,
               'min_samples_leaf': min_samples_leaf}
print(random_grid)


# In[49]:


#now we use random grid for best parameters
rf_random= RandomizedSearchCV(estimator= rf_regressor, param_distributions= random_grid,scoring = 'neg_mean_squared_error', 
                             n_iter = 10, cv = 5, verbose = 2, random_state = 42, n_jobs= 1)


# In[50]:


rf_random.fit(x_train, y_train)


# In[51]:


rf_random.best_params_


# In[52]:


rf_random.best_score_


# In[53]:


#now predict the model
prediction = rf_random.predict(x_test)


# In[54]:


sns.distplot(y_test-prediction)


# In[55]:


#From above we can see that the difference gives the normal distribution which means model give good predictions


# In[56]:


sns.scatterplot(y_test, prediction)


# In[57]:


#we can see that we get linear data points


# In[58]:


from sklearn import metrics
print('MAE:', metrics.mean_absolute_error(y_test, prediction))
print('MSE:', metrics.mean_squared_error(y_test, prediction))
print('RMSE:', np.sqrt(metrics.mean_squared_error(y_test, prediction)))


# In[59]:


import pickle
# open a file, where you ant to store the data
file = open("random_forest_regression_model.pkl", 'wb')
#dump information to that file
pickle.dump(rf_random, file)


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




