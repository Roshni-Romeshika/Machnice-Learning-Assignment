#!/usr/bin/env python
# coding: utf-8

# In[67]:


# IT17100076 Notebook created.
# Importing Python Libraries

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

get_ipython().run_line_magic('matplotlib', 'inline')
sns.set_style("darkgrid")
plt.style.use("dark_background")


# In[68]:


# Loading data

dataframe = pd.read_csv("D:\\SLIIT\\4th Year\\1st Semester\\ML\\Assignments\\Assignment - 01\\IT17100076\\it17100076_heart_disease_dataset.csv")


# In[69]:


# All data of first 5 rows

dataframe.head()


# In[70]:


# Data Analysis

display(dataframe.info(), dataframe.describe(), dataframe.shape)


# In[71]:


# checking for null values

dataframe.isna().sum()


# In[72]:


# Getting target count

colors = ['darkturquoise', 'salmon']
plt.style.use('dark_background')
plt.rcParams['figure.figsize']=(9,8)

axis = sns.countplot(x='target', data=dataframe, palette=colors, alpha=0.9, edgecolor=('white'), linewidth=4)
axis.set_ylabel('count', fontsize=12)
axis.set_xlabel('target', fontsize=12)
axis.grid(b=True, which='major', color='grey', linewidth=0.2)
plt.title('Target count', fontsize=15)
plt.show()

target_0 = len(dataframe[dataframe.target == 0])
target_1 = len(dataframe[dataframe.target == 1])

print("Percentage of negative Heart Disease: {:.2f}%".format((target_0 / (len(dataframe.target))*100)))
print("Percentage of  positive Heart Disease: {:.2f}%".format((target_1 / (len(dataframe.target))*100)))
dataframe.target.value_counts()


# In[73]:


# Creating correlation metrix

plt.rcParams['figure.figsize'] = 15, 15
plt.style.use('dark_background')
plt.matshow(dataframe.corr()) 
plt.yticks(np.arange(dataframe.shape[1]), dataframe.columns) 
plt.xticks(np.arange(dataframe.shape[1]), dataframe.columns) 
plt.colorbar()


# In[74]:


# Correlation matrix

plt.style.use('dark_background')
f, (axis1, axis2) = plt.subplots(1,2,figsize =(15, 8))
corr = dataframe.corr()
mask = np.zeros_like(corr, dtype=np.bool)
mask[np.triu_indices_from(mask)] = True
heatmapkws = dict(linewidths=0.1) 

sns.heatmap((dataframe[dataframe['target'] ==1]).corr(), vmax = .8, square=True, ax = axis1, cmap = 'YlGnBu', mask=mask, **heatmapkws);
sns.heatmap((dataframe[dataframe['target'] ==0]).corr(), vmax = .8, square=True, ax = axis2, cmap = 'afmhot', mask=mask,**heatmapkws);

axis1.set_title('Healthy Chart', fontsize=14)
axis2.set_title('Disease Chart', fontsize=14)

plt.show()


# In[75]:


dataframe.hist() 
plt.style.use('dark_background')


# In[76]:


# Create another figure to expose max heart rate for age
plt.figure(figsize=(10, 8))

# Scatter with negative and postivie examples

plt.scatter(dataframe.age[dataframe.target==0],
            dataframe.thalach[dataframe.target==0],
            c="lightblue")

plt.scatter(dataframe.age[dataframe.target==1],
            dataframe.thalach[dataframe.target==1],
            c="salmon")




plt.title("Heart Disease in function in range of Age and Max Heart Rate")
plt.xlabel("Age")
plt.ylabel("Max Heart Rate")
plt.legend(["positive", "Negative"]);


# In[77]:


# Filtering values

categorical_values = []
continous_values = []
for column in dataframe.columns:
    if len(dataframe[column].unique()) <= 10:
        categorical_values.append(column)
    else:
        continous_values.append(column)


# In[78]:


# Getting categorical values from the dataset

categorical_values


# In[79]:


# Creating dummy columns for categorical values

categorical_values.remove('target')
dataset = pd.get_dummies(dataframe, columns = categorical_values)
dataset.head()


# In[80]:


# Scalling provied columns
from sklearn.preprocessing import StandardScaler

ssc = StandardScaler()
columns_to_scale = ['age', 'trestbps', 'chol', 'thalach', 'oldpeak']
dataset[columns_to_scale] = ssc.fit_transform(dataset[columns_to_scale])


# In[81]:


dataset.head()


# In[82]:


# Applying machine learning algorithm
# Data processing modeling and metrics

from sklearn.metrics import accuracy_score, confusion_matrix, precision_score, recall_score, f1_score


def print_score(clf, X_trainingdataset, y_trainingdataset, X_testingdataset, y_testingdataset, train=True):
    if train:
        pred = clf.predict(X_trainingdataset)
        print("Trained Result:\n********************************************")
        print(f"Accuracy Score: {accuracy_score(y_trainingdataset, pred) * 100:.2f}%")
        print("_______________________________________________")
        print("Classification Report for trained dataset:", end='')
        print(f"\tPrecision Score: {precision_score(y_trainingdataset, pred) * 100:.2f}%")
        print(f"\t\t\t\t\t\tRecall Score: {recall_score(y_trainingdataset, pred) * 100:.2f}%")
        print(f"\t\t\t\t\t\tF1 score: {f1_score(y_trainingdataset, pred) * 100:.2f}%")
        print("_______________________________________________")
        print(f"Confusion Matrix: \n {confusion_matrix(y_trainingdataset, pred)}\n")
        
    elif train==False:
        pred = clf.predict(X_testingdataset)
        print("\n\n\nTest Result:\n*******************************************")        
        print(f"Accuracy Score: {accuracy_score(y_testingdataset, pred) * 100:.2f}%")
        print("_______________________________________________")
        print("Classification Report for test dataset:", end=' ')
        print(f"\tPrecision Score: {precision_score(y_testingdataset, pred) * 100:.2f}%")
        print(f"\t\t\t\t\t\tRecall Score: {recall_score(y_testingdataset, pred) * 100:.2f}%")
        print(f"\t\t\t\t\t\tF1 score: {f1_score(y_testingdataset, pred) * 100:.2f}%")
        print("_______________________________________________")
        print(f"Confusion Matrix: \n {confusion_matrix(y_testingdataset, pred)}\n")


# In[83]:


#Split dataset into 2 separte datasets
from sklearn.model_selection import train_test_split

X = dataset.drop('target', axis=1)
y = dataset.target

X_trainingdataset, X_testingdataset, y_trainingdataset, y_testingdataset = train_test_split(X, y, test_size=0.3, random_state=42)


# In[84]:


# Apply Random Forest Classifier Algorithm

from sklearn.ensemble import RandomForestClassifier


random_forest = RandomForestClassifier(n_estimators=1000, random_state=42)
random_forest.fit(X_trainingdataset, y_trainingdataset)


print_score(random_forest, X_trainingdataset, y_trainingdataset, X_testingdataset, y_testingdataset, train=True)
print_score(random_forest, X_trainingdataset, y_trainingdataset, X_testingdataset, y_testingdataset, train=False)


# In[85]:


# Finalize the training and testing accuracy in Random Forest

test_score = accuracy_score(y_testingdataset, random_forest.predict(X_testingdataset)) * 100
train_score = accuracy_score(y_trainingdataset, random_forest.predict(X_trainingdataset)) * 100

results_dataframe = pd.DataFrame(data=[["Random Forest Classifier", train_score, test_score]], 
                          columns=['Model', 'Training Accuracy %', 'Testing Accuracy %'])
results_dataframe


# In[86]:


feature_importance = pd.DataFrame({'feature': list(X_trainingdataset.columns),
                   'importance': random_forest.feature_importances_}).\
                    sort_values('importance', ascending = False)

# Display importance of feature 
feature_importance.head()


# In[ ]:




