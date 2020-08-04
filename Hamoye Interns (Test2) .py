#!/usr/bin/env python
# coding: utf-8

# In[2]:


import numpy as np
import pandas as pd
import seaborn as sns
from sklearn import metrics
import matplotlib.pyplot as plt
import seaborn as seabornInstance
get_ipython().run_line_magic('matplotlib', 'inline')

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import MinMaxScaler
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error
from sklearn.linear_model import Lasso


# In[6]:


me = pd.read_csv('Downloads\energydata_complete.csv')


# In[7]:


me


# In[9]:


simple_linear_reg_df = me[["date", 'lights']].sample(15, random_state=2)


# In[11]:


sns.regplot(x="date", y="lights", data=simple_linear_reg_df)


# In[10]:


simple_linear_reg_df


# In[42]:


simple_linear_reg_me = me[["T2", 'T6']].sample(15, random_state=2)


# In[43]:


simple_linear_reg_me


# In[44]:


sns.regplot(x="T2", y="T6", data=simple_linear_reg_me)


# In[12]:


features_me = me.drop(columns=['date', 'lights'])


# In[56]:


another_me = features_me.drop(columns=['T2','T6'])


# In[57]:


another_me


# In[55]:


features_me


# In[50]:


scaler = MinMaxScaler()


# In[51]:


normalised_ans = pd.DataFrame(scaler.fit_transform(another_me), columns=another_me.columns)


# In[ ]:





# In[16]:


normalised_me = pd.DataFrame(scaler.fit_transform(features_me), columns=features_me.columns)


# In[17]:


normalised_me


# In[18]:


appliance_target = normalised_me['Appliances']


# In[46]:


T2_target = normalised_me['T2']


# In[47]:


T2_target


# In[19]:


appliance_target


# In[58]:


x_train, x_test, y_train, y_test = train_test_split(another_me, T2_target, test_size=0.3, random_state=42)


# In[59]:


linear_model = LinearRegression()


# In[60]:


linear_model.fit(x_train, y_train)


# In[61]:


predicted_values = linear_model.predict(x_test)


# In[62]:


predicted_values


# In[63]:


mae = mean_absolute_error(y_test, predicted_values)


# In[64]:


mae


# In[65]:


round(mae, 2)


# In[27]:


round(mae, 3)


# In[66]:


from sklearn.metrics import r2_score
r2_score = r2_score(y_test, predicted_values)
r2_score


# In[69]:


from sklearn.metrics import r2_score
r2_score = r2_score(y_test, predicted_values)
round(r2_score, 2)


# In[70]:


rss = np.sum(np.square(y_test - predicted_values))
round(rss, 2)


# In[72]:


from sklearn.metrics import  mean_squared_error
rmse = np.sqrt(mean_squared_error(y_test, predicted_values))
round(rmse, 3)


# In[73]:


def get_weights_me(model, feat, col_name):
  weights = pd.Series(model.coef_, feat.columns).sort_values()
  weights_me = pd.DataFrame(weights).reset_index()
  weights_me.columns = ['Features', col_name]
  weights_me[col_name].round(3)
  return weights_me


# In[80]:


from sklearn.linear_model import Lasso
lasso_reg = Lasso(alpha=0.001)
lasso_reg.fit(x_train, y_train)


# In[75]:


from sklearn.linear_model import Ridge
ridge_reg = Ridge(alpha=0.5)
ridge_reg.fit(x_train, y_train)


# In[76]:


linear_model_weights = get_weights_me(linear_model, x_train, 'Linear_Model_Weight')
ridge_weights_me = get_weights_me(ridge_reg, x_train, 'Ridge_Weight')
lasso_weights_me = get_weights_me(lasso_reg, x_train, 'Lasso_weight')


# In[81]:


lasso_weights_me


# In[77]:


final_weights = pd.merge(linear_model_weights, ridge_weights_me, on='Features')
final_weights = pd.merge(final_weights, lasso_weights_me, on='Features')


# In[78]:


final_weights


# In[84]:


round(lasso_weights_me, 3)


# In[ ]:




