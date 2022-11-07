#!/usr/bin/env python
# coding: utf-8

# In[1]:


# Question 1 - Measures of Center

import pandas as pd

# read in the file mtcars.csv
df = pd.read_csv("mtcars.csv")

# find the mean of the column wt
mean = df["wt"].mean()

# find the median of the column wt
median = df["wt"].median()

# find the mode of the column wt
mode = df["wt"].mode()

print("mean = ", mean, ", median = ", median, ", mode = ", mode)


# In[2]:


# Question 2 - Standard Deviation

import pandas as pd
import numpy as np

# Also import the scipy.stats module.
import scipy.stats as st

# Type your code here to load the csv file NBA2019.csv
NBA2019_df = pd.read_csv("NBA2019.csv")

# Input desired column. Ex: AGE, 2P%, or PointsPerGame.
chosen_column = str(input("What column do you need the standard deviation for? "))

# Create subset of NBA2019_df based on input.
NBA2019_df_column = NBA2019_df[[chosen_column]]

# Find standard deviation and round to two decimal places. 
sample_s = st.tstd(NBA2019_df_column)
sample_s_rounded = np.round(sample_s, 2)

sample_s_rounded = str(sample_s_rounded).replace('[','').replace(']','').replace('\'','').replace('\"','')

# Output
print('The standard deviation for', chosen_column, "is:", sample_s_rounded)


# In[3]:


# Question 3 - Five Number Summary

import pandas as pd

df = pd.read_csv("internetusage.csv")

# subset the column internet_usage
internet = df[["internet_usage"]]

# find the five number summary
five_num = internet.describe()

print(five_num)


# In[4]:


# Question 4 - Box Plots

# import the pandas module as pd
import pandas as pd

# load internetusage.csv 
df = pd.read_csv("internetusage.csv ")

# subset the State and Population columns
population = df[["State", "Population"]]

state_index = str(input("Enter a state to find the population: "))

# subset the row given by state_index
state_data = population[population["State"] == state_index]

state_name = state_data.iloc[0][0]
state_pop = state_data.iloc[0][1]

print("The population of " + str(state_name) + " is " + str(state_pop)+ ".")


# In[5]:


# Create a box plot for Percentage Internet Usage in the US

import matplotlib.pyplot as plt
import seaborn as sns

sns.set(style="darkgrid", color_codes=True)
ax = plt.subplots(figsize=(9,6))

plt.title("Box Plot for Percentage Internet Usage in the US (Excluding Territories)", fontsize=18)
plt.xlabel("Internet Usage", fontsize=14)

ax = sns.boxplot(y="internet_usage", color="lightskyblue", data=df)
ax = sns.stripplot(y="internet_usage", color="blue", size=7, data=df)
plt.ylabel("")
plt.tight_layout

plt.show()


# In[6]:


# Create a box plot for the population data frame

import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import seaborn as sns

sns.set(style="darkgrid", color_codes=True)
ax = plt.subplots(figsize=(9,5))

plt.title('Box Plot for State Populations in the US (Millions Excluding Territories)', fontsize=18)
plt.xlabel("Population", fontsize=14)

ax = sns.boxplot(y="Population", color="lightskyblue", width=.6, data=population)
ax = sns.swarmplot(y="Population", color="blue", size=7, data=population)

plt.ylabel("")
plt.tight_layout

ax.yaxis.set_major_formatter(ticker.FuncFormatter(lambda x, pos: '{:,.0f}'.format(x/1000000) + 'M'))

plt.show()


# In[9]:


# Create an Interactive box plot for Percentage Internet Usage in the US

import plotly.express as px
import plotly

fig = px.box(df, y="internet_usage", custom_data=["State"], points="all", labels={"internet_usage": "Internet Usage"}, hover_data=['internet_usage', 'State'])

fig.update_layout(xaxis_title="Internet Usage", yaxis_title="", title="Box Plot for Percentage Internet Usage in the US (Excluding Territories)")

fig.update_traces(hovertemplate = "(Internet Usage, %{y}) <br> State: %{customdata} <br> Internet Usage: %{y}%")

fig.show()


# In[10]:


# Create an Interactive box plot for the Population data frame

import plotly.express as px
import plotly

fig = px.box(df, y="Population", custom_data=["State"], points="all", labels={"Population": "Population"}, hover_data=['Population', 'State'])

fig.update_layout(xaxis_title="Population", yaxis_title="", title="Box Plot for State Populations in the US (Millions Excluding Territories)")

fig.update_traces(hovertemplate = "(Population, %{y}) <br> %{customdata}")

fig.show()


# In[ ]:




