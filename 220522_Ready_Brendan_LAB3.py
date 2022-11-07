#!/usr/bin/env python
# coding: utf-8

# In[1]:


# Question 1 - Creating SLR Models

# Import the necessary modules
import pandas as pd
import statsmodels.api as sm
import statsmodels.formula.api as sms

# Code to read in nbaallelo_slr.csv
nba = pd.read_csv('nbaallelo_slr.csv')

# Create a new column in the data frame that is the difference between pts and opp_pts
# Code to find the difference between the columns pts and opp_pts
nba['y'] = nba['pts'] - nba['opp_pts']

# Perform simple linear regression on y and elo_n
# Code to perform SLR using statsmodels ols
results = sms.ols('y~elo_n', data=nba).fit()

# Create an analysis of variance table
# Code to create ANOVA table , use Type 2 since no interaction effect is assumed between the independent variables
aov_table = sm.stats.anova_lm(results, typ=2)

# Print the analysis of variance table
print(aov_table)

print('\n')

# Perform simple linear regression on y and elo_i
# Code to perform SLR using statsmodels ols
results = sms.ols('y~elo_i', data=nba).fit()

# Create an analysis of variance table
# Code to create ANOVA table , use Type 2 since no interaction effect is assumed between the independent variables
aov_table = sm.stats.anova_lm(results, typ=2)

# Print the analysis of variance table
print(aov_table)


# In[2]:


# Question 2 - Making predictions using SLR Models

import numpy as np
import pandas as pd
import statsmodels.formula.api as sms

# load the file internetusage.csv
internet = pd.read_csv('internetusage.csv')

# fit a linear model using the sms.ols function and the internet dataframe
model = sms.ols('internet_usage~bachelors_degree', data=internet).fit()

bach_percent = float(input("Enter the percentage of individuals with bachelor's degrees: "))

# use the model.predict function to find the predicted value for internet_usage using 
# the bach_percent value for the predictor
prediction = model.predict(exog = {'bachelors_degree':bach_percent})

print("The predicted percentage of internet users in a state is: \n", prediction)


# In[3]:


# Question 3 - Creating Correlation Matrices

#Import the necessary modules
import pandas as pd

# Code to read in nbaallelo_slr.csv
nba = pd.read_csv('nbaallelo_slr.csv')

# Display the correlation matrix for the columns elo_n, pts, and opp_pts
# Code to calculate correlation matrix
#print(nba.corr())
print(nba[['elo_n', 'pts', 'opp_pts']].corr())

# Create a new column in the data frame that is the difference between pts and opp_pts
# Code to find the difference between the columns pts and opp_pts
nba['y'] = nba['pts'] - nba['opp_pts']

# Display the correlation matrix for elo_n and y
# Code to calculate the correlation matrix
print(nba[['elo_n', 'y']].corr())

print('\n')

# Display the correlation matrix for the columns elo_i, pts, and opp_pts
# Code to calculate correlation matrix
#print(nba.corr())
print(nba[['elo_i', 'pts', 'opp_pts']].corr())

# Create a new column in the data frame that is the difference between pts and opp_pts
# Code to find the difference between the columns pts and opp_pts
nba['y'] = nba['pts'] - nba['opp_pts']

# Display the correlation matrix for elo_i and y
# Code to calculate the correlation matrix
print(nba[['elo_i', 'y']].corr())


# In[4]:


# Question 4 - Multiple Regression

# Import the necessary modules
import pandas as pd
import statsmodels.api as sm
import statsmodels.formula.api as sms

# Code to read in nbaallelo_slr.csv
nba = pd.read_csv('nbaallelo_slr.csv')

# Perform multiple linear regression on pts, elo_n, and opp_pts
# Code to perform multiple regression using statsmodels ols 
results = sms.ols('pts~elo_n + opp_pts', data=nba).fit()

# Create an analysis of variance table
# Code to create ANOVA table
aov_table = sm.stats.anova_lm(results, typ=2)

# Print the analysis of variance table
print(aov_table)

print('\n')

# Perform multiple linear regression on pts, elo_i, and opp_pts
# Code to perform multiple regression using statsmodels ols 
results = sms.ols('pts~elo_i + opp_pts', data=nba).fit()

# Create an analysis of variance table
# Code to create ANOVA table
aov_table = sm.stats.anova_lm(results, typ=2)

# Print the analysis of variance table
print(aov_table)


# In[ ]:




