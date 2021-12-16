#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report


# In[2]:


# So that we may see all columns
pd.set_option('display.max_columns', None)


# In[3]:


# Read in the data
data = pd.read_csv("weatherAUS.csv")

# Convert 9pm/3pm data values to "delta" values and drop the originals from the dataframe
data["deltaWindSpeed"] = data["WindSpeed3pm"] - data["WindSpeed9am"]
data["deltaHumidity"] = data["Humidity3pm"] - data["Humidity9am"]
data["deltaPressure"] = data["Pressure3pm"] - data["Pressure9am"]
data["deltaCloud"] = data["Cloud3pm"] - data["Cloud9am"]
data["deltaTemp"] = data["Temp3pm"] - data["Temp9am"]
data["RainToday"] = np.where(data["RainToday"] == "Yes", 1, 0)
data["RainTomorrow"] = np.where(data["RainTomorrow"] == "Yes", 1, 0)
data = data.drop(
    ["Location", "WindSpeed9am", "WindSpeed3pm", "Humidity9am", "Humidity3pm", "Pressure9am", "Pressure3pm",
       "Cloud9am", "Cloud3pm", "Temp9am", "Temp3pm"], axis=1)

data = data.interpolate()

# http: // www.land - navigation.com / boxing - the - compass.html
d = {'N': 0, 'NNE': 1, 'NE': 2, 'ENE': 3, 'E': 4, 'ESE': 5, 'SE': 6, 'SSE': 7, 'S': 8, 'SSW': 9, 'SW': 10,
        'WSW': 11, 'W': 12, 'WNW': 13, 'NW': 14, 'NNW': 15}

data['WindGustDir'] = data['WindGustDir'].map(d)
data['WindDir9am'] = data['WindDir9am'].map(d)
data['WindDir3pm'] = data['WindDir3pm'].map(d)

data['WindDirDelta'] = (data['WindDir3pm'] - data['WindDir9am']) % 16

# Remove rows with NA values from the data
dataNoNA = data.dropna()

dateData = pd.to_datetime(dataNoNA['Date'])
dataNoNA = dataNoNA.drop(["Date"], axis=1)

# Separate labels from the data
labels = dataNoNA["RainTomorrow"]
dataNoNA = dataNoNA.drop(["RainTomorrow"], axis=1)

# Standardization
dataNoNA = (dataNoNA - dataNoNA.mean()) / dataNoNA.std()

dataNoNA["Date"] = dateData

# Train-test split (80-20)
x_train, x_test, y_train, y_test = train_test_split(dataNoNA, labels, test_size=0.2, random_state=479)


# In[4]:


x_train.dtypes


# In[5]:


x_train["Year"] = x_train["Date"].dt.year           #abstracting year in different column 
x_train["Month"] = x_train["Date"].dt.month         #abstracting month in diffrent column  
x_train["Day"] = x_train["Date"].dt.day             #abstracting day in diffrent column  



x_test["Year"] = x_test["Date"].dt.year           #abstracting year in different column 
x_test["Month"] = x_test["Date"].dt.month         #abstracting month in diffrent column  
x_test["Day"] = x_test["Date"].dt.day             #abstracting day in diffrent column  


# In[6]:


x_train.drop("Date", axis = 1, inplace = True)
x_test.drop("Date", axis = 1, inplace = True)


# In[7]:


x_train.head()


# In[8]:


x_train1, x_dev, y_train1, y_dev = train_test_split(x_train, y_train, test_size=0.2, random_state=479)


# In[9]:


#SVM
from sklearn.svm import SVC

svm = SVC(kernel ='linear')


svm.fit(x_train1, y_train1)


# In[13]:


from sklearn.model_selection import GridSearchCV

param_grid = {'C': [0.001,0.01,0.1,1,10,100,1000], 'gamma': [1,0.1,0.01,0.001,0.0001], 'kernel': ['linear']} 
grid = GridSearchCV(SVC(),param_grid,refit=True,verbose=3

#5-fold CV on development data
grid.fit(x_dev,y_dev)


# In[14]:


grid.best_params_


# In[15]:


grid.best_score_


# In[16]:


#re-train with best parameters
svm = SVC(C=100, gamma =1, kernel ='linear')

#retrain original x_train, y_train data
svm.fit(x_train, y_train)


# In[21]:


predictions = svm.predict(x_test)

print(svm.score(x_test, y_test))


# In[19]:


print(classification_report(y_test, predictions))


# In[ ]:




