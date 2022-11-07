#!/usr/bin/env python
# coding: utf-8

# In[1]:


# LAB 4 - Introduction to Data Mining, Data Cleansing and Preparation
# Question 1 - dropna() and fillna()

# Import the necessary modules
import pandas as pd

# Code to read in hmeq_small.csv
hmeq = pd.read_csv("hmeq_small.csv")

# Create a new data frame with the rows with missing values dropped
# Code to delete rows with missing values
df1 = hmeq.dropna()

# Create a new data frame with the missing values filled in by the mean of the column
# Code to fill in missing values
df2 = hmeq.fillna(value=hmeq.mean())

# Print the means of the columns for each new data frame
# Code to find means of df1
print("Means for df1 are ", df1.mean())

# Code to find means of df2
print("Means for df2 are ", df2.mean())


# In[2]:


# LAB 4 - Introduction to Data Mining, Data Cleansing and Preparation
# Question 2 - scale() and MinMaxScaler()

# Import the necessary modules
import pandas as pd
from sklearn import preprocessing

# Read in the file hmeq_small.csv
hmeq = pd.read_csv("hmeq_lab412.csv")

# Standardize the data
# Code to standardize the data
standardized = preprocessing.scale(hmeq)

# Output the standardized data as a data frame
# Code to output as a data frame
df1 = pd.DataFrame(standardized, columns = ["LOAN","MORTDUE","VALUE","YOJ","CLAGE","CLNO","DEBTINC"])

# Normalize the data
# Code to normalize the data
normalized = preprocessing.MinMaxScaler().fit_transform(hmeq)

# Output the standardized data as a data frame
# Code to output as a data frame
df2 = pd.DataFrame(normalized, columns = ["LOAN","MORTDUE","VALUE","YOJ","CLAGE","CLNO","DEBTINC"])

# Print the means and standard deviations of df1 and df2
# Code for mean of df1
print("The means of df1 are ", df1.mean())

# Code for standard deviation of df1
print("The standard deviations of df1 are ", df1.std())

# Code for mean of df2
print("The means of df2 are ", df2.mean())

# Code for standard deviation of df2
print("The standard deviations of df2 are ", df2.std())


# In[ ]:




