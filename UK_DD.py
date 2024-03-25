#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd


# In[2]:


#create data
df = pd.DataFrame({'cases': [859, 2360, 1512, 846, 209, 674, 1651, 2083, 3973, 3214, 3338],
                  'hospitalizations': [2506, 4312, 2535, 1804, 721, 1867, 3143, 4958, 6187, 4826, 7156],
                  'deaths': [45, 42, 20, 16, 9, 29, 44, 60, 63, 61, 111]})


# In[3]:


#view data
df


# In[4]:


import statsmodels.api as sm


# In[5]:


#define response variable
y = df['deaths']


# In[6]:


#define predictor variable
x = df[['cases', 'hospitalizations']]


# In[7]:


#add constant to predictor variables
x = sm.add_constant(x)


# In[8]:


#fit linear regression model
model = sm.OLS(y, x).fit()


# In[9]:


#view model summary
print(model.summary())


# In[11]:


#import necessary libraries
import matplotlib.pyplot as plt
import statsmodels.api as sm
from statsmodels.formula.api import ols


# In[12]:


#fit simple linear regression model 
model = ols('cases ~ deaths', data=df).fit()


# In[13]:


#view model summary
print(model.summary())


# In[14]:


#define figure size
fig = plt.figure(figsize=(12,8))

#produce regression plots
fig = sm.graphics.plot_regress_exog(model,'deaths', fig=fig)


# In[15]:


-6.2159 - 0.0176*4000 + 0.0233*8000


# In[16]:


-6.2159 - 0.0176*5000 + 0.0233*9000


# In[17]:


-6.2159 - 0.0176*6000 + 0.0233*10000


# In[ ]:




