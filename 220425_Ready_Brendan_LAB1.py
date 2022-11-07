#!/usr/bin/env python
# coding: utf-8

# In[1]:


# LAB 1 - Data Visualization
# Question 1 - Importing Modules

# add code to import numpy and scipy
import numpy as np
import scipy.stats as st

x1 = int(input())
x2 = int(input())
x3 = int(input())
x4 = int(input())

x = np.array([x1, x2, x3, x4])
y = np.array([0, 10, 7, 25])

print(st.linregress(x,y))


# In[2]:


# LAB 1 - Data Visualization
# Question 2 - Data Frames

import pandas as pd

# Import the CSV file Cars.csv
cars_df = pd.read_csv("cars.csv")

userNum = int(input())

# Subset the first userNum rows of the data frame
car_df = cars_df[cars_df.Quality == userNum]

# add code to find maximum values of the subset
print(car_df.max())


# In[3]:


# LAB 1 - Data Visualization
# Question 3 - Subsetting Data Frames

import pandas as pd
cylinders = int(input())

# read in the csv file mtcars.csv
df = pd.read_csv("mtcars.csv")

# create a new dataframe with only the rows where cyl = cylinders
df_cyl = df[df.cyl == cylinders]

# print the shape of the new data frame
print(df_cyl)


# In[5]:


# LAB 1 - Data Visualization
# Question 4 - Bar Charts

# load the pandas module as pd
import pandas as pd

# load titanic.csv
titanic = pd.read_csv("titanic.csv")

# subset the titanic dataset to include first class passengers who embarked in Southampton
first_south = titanic[(titanic["pclass"] == 1) & (titanic["embarked"] == "S")]

# subset the titanic dataset to include either second or third class passengers
second_third = titanic[titanic["pclass"].isin([2,3])]

print(first_south.head())
print(second_third.head())


# In[6]:


# LAB 1 - Data Visualization
# Question 4 - Bar Charts

# Using the template main.py, create and upload the bar charts:
import matplotlib.pyplot as plt
import seaborn as sns

#1 show passengers in first, second, and third class grouped by sex

# style bar chart
sns.set(style="darkgrid", color_codes=True)

# set title
plt.title('Passenger Class', fontsize=15)

# set colors for bar chart
clrs = ['red', 'blue']

# plot vertical bar chart
sns.countplot(x="class", hue="sex", palette=clrs, order = ["First", "Second", "Third"], data=titanic);

# show bar chart
plt.show()

#2 show passengers in first class who embarked in Southampton grouped by sex

# style bar chart
sns.set(style="darkgrid", color_codes=True)

# set colors for bar chart
clrs = ['blue', 'red']

# set title
plt.title('Passenger Class', fontsize=15)

# plot vertical bar chart
sns.countplot(x="class", hue="sex", palette=clrs, data=first_south);

# show bar chart
plt.show()

#3 show passengers in first, second, and third class grouped by survival status

# style bar chart
sns.set(style="darkgrid", color_codes=True)

# set title
plt.title('Passenger Class', fontsize=15)

# set colors for bar chart
clrs = ['purple', 'green']

# plot vertical bar chart
sns.countplot(x="class", hue="survived", palette=clrs, order = ["First", "Second", "Third"], data=titanic);

plt.legend(labels=["Not Survived", "Survived"])

# show bar chart
plt.show()

#4 show passengers in second and third class grouped by survival status

second_third = titanic[titanic["pclass"].isin([2,3])]

# sets the style of the bar charts
sns.set(style="darkgrid", color_codes=True)

# title
plt.title('Passenger Class', fontsize=15)

# set colors for bar chart
clrs = ['purple', 'green']

# generates a vertical bar chart
sns.countplot(x="class", hue="survived", palette=clrs, data=second_third, order = ["Second", "Third"]);

plt.legend(labels=["Not Survived", "Survived"])     

# shows the image
plt.show()


# In[7]:


# LAB 1 - Data Visualization
# Question 4 - Bar Charts

# Create an Interactive bar chart that shows passengers in first, second, and third class grouped by sex

import plotly.express as px
import plotly

fig = px.histogram(titanic, x="class", color="sex", labels={"sex": ""}, histfunc='count', barmode='group', color_discrete_map={'male': 'red','female': 'blue'})

fig.update_layout(xaxis_title="Passenger Class", yaxis_title="Count", title="Total Passengers grouped by sex", xaxis={'categoryorder':'category ascending'})

fig.update_traces(hovertemplate = "(%{x}, %{y})")

fig.show()


# In[8]:


# LAB 1 - Data Visualization
# Question 4 - Bar Charts

# Create an Interactive bar chart that shows passengers in first class who embarked in Southampton grouped by sex

fig = px.histogram(first_south, x="class", color="sex", labels={"sex": ""}, histfunc='count', barmode='group', color_discrete_map={'male': 'red','female': 'blue'})

fig.update_layout(xaxis_title="Passenger Class", yaxis_title="Count", title="First class Passengers who embarked in Southampton grouped by sex")

fig.update_traces(hovertemplate = "(%{x}, %{y})")

fig.show()


# In[10]:


# LAB 1 - Data Visualization
# Question 4 - Bar Charts

# Create an Interactive bar chart that shows passengers in first, second, and third class grouped by survival status

def custom_legend_name(new_names):
    for i, new_name in enumerate(new_names):
        fig.data[i].name = new_name
        
fig = px.histogram(titanic, x="class", color="survived", labels={"survived": ""}, histfunc='count', barmode='group', color_discrete_map={0: 'purple', 1: 'green'})

fig.update_layout(xaxis_title="Passenger Class", yaxis_title="Count", title="Total Passengers grouped by survival status", xaxis={'categoryorder':'category ascending'})

fig.update_traces(hovertemplate = "(%{x}, %{y})")

custom_legend_name(['Not Survived','Survived'])

fig.show()


# In[12]:


# LAB 1 - Data Visualization
# Question 4 - Bar Charts

# Create an Interactive bar chart that shows passengers in second and third class grouped by survival status

second_third = titanic[titanic["pclass"].isin([2,3])]

def custom_legend_name(new_names):
    for i, new_name in enumerate(new_names):
        fig.data[i].name = new_name
        
fig = px.histogram(second_third, x="class", color="survived", labels={"survived": ""}, histfunc='count', barmode='group', color_discrete_map={0: 'purple', 1: 'green'})

fig.update_layout(xaxis_title="Passenger Class", yaxis_title="Count", title="Second and Third class Passengers grouped by survival status", xaxis={'categoryorder':'category ascending'})

fig.update_traces(hovertemplate = "(%{x}, %{y})")

custom_legend_name(['Not Survived','Survived'])

fig.show()


# In[14]:


# LAB 1 - Data Visualization
# Question 5 - Line Charts

# import the pandas module
import pandas as pd
import datetime

# load the target.csv file
tgt = pd.read_csv("target.csv")

# subset the last 19 days of the dataframe
tgt_march = tgt[-19:]

# subset tgt_march and create a data frame that contains the columns: Date and Volume
tgt_vol = tgt_march[["Date","Volume"]]

# subset tgt_march and create a data frame that contains the columns: Date and Closing
tgt_close = tgt_march[["Date","Close"]]

day = int(input())

# subset the specific row of tgt_vol for the given day
volume_row = tgt_vol[pd.to_datetime(tgt_vol['Date']).dt.strftime('%d') == str(day)]

# gets the volume for the given day
volume = volume_row.iloc[0][1] 

# subset the specific row of tgt_close for the given day
close_row = tgt_close[pd.to_datetime(tgt_close['Date']).dt.strftime('%d') == str(day)]

# gets the closing stock price for the given day
close = close_row.iloc[0][1] 

# gets the date
date = tgt_march[pd.to_datetime(tgt_march['Date']).dt.strftime('%d') == str(day)].iloc[0][0] 

print("The volume of TGT on", date, "is", volume)
print("The closing stock price of TGT on", date, "is $", close)


# In[16]:


# LAB 1 - Data Visualization
# Question 5 - Line Charts

# Using the template from main.py, create separate line charts for Volume and Close

import matplotlib.pyplot as plt

# 1 Show line chart for Volume

# set title
plt.title('March 2018 Trading Volume for Target Stock', fontsize = 15)

# set x and y axis labels
plt.xlabel('Date')
plt.ylabel('Number of trades')

# plot line chart
plt.plot(pd.to_datetime(tgt_march["Date"]).dt.strftime('%d'), tgt_march["Volume"], "black")

# show line chart
plt.show()

# 2 Show line chart for Close

# set title
plt.title('March 2018 Market Closing Price for Target Stock', fontsize = 15)

# set x and y axis labels
plt.xlabel('Date')
plt.ylabel('Price')

# plot line chart
plt.plot(pd.to_datetime(tgt_march["Date"]).dt.strftime('%d'), tgt_march["Close"], "r-")

# show line chart
plt.show()


# In[17]:


# LAB 1 - Data Visualization
# Question 5 - Line Charts

# Create an interactive line chart for Volume

import plotly.express as px

marchdate = pd.to_datetime(tgt_march["Date"])
marchvolume = tgt_march["Volume"]

fig = px.line(tgt_march, x=marchdate, y=marchvolume, labels={"x": "Date", "Volume": "Number of Trades"}, title='March 2018 Trading Volume for Target Stock')
fig.update_traces(line_color='black')
fig['data'][0]['showlegend']=True
fig['data'][0]['name']='Volume'
fig.show()


# In[18]:


# LAB 1 - Data Visualization
# Question 5 - Line Charts

# Create an interactive line chart for Close

marchdate = pd.to_datetime(tgt_march["Date"])
marchclose = tgt_march["Close"]

fig = px.line(tgt_march, x=marchdate, y=marchclose, labels={"x": "Date", "Close": "Price"}, title='March 2018 Market Closing Price for Target Stock')
fig.update_traces(line_color='red')
fig['data'][0]['showlegend']=True
fig['data'][0]['name']='Price'
fig.show()


# In[19]:


# LAB 1 - Data Visualization
# Question 6 - Strip Plots

# load pandas module as pd
import pandas as pd

# load the titanic.csv dataset
titanic = pd.read_csv("titanic.csv")

# subset titanic to include male passengers in first class over 18 years old
df = titanic[(titanic["sex"] == "male") & (titanic["pclass"] == 1) & (titanic["age"] > 18)]

print(df.head())


# In[20]:


# LAB 1 - Data Visualization
# Question 6 - Strip Plots

# Create a strip plot where the data is grouped by the city the passengers embarked and by survival status

import matplotlib.pyplot as plt
import seaborn as sns

# set the style of the strip plot
sns.set(style="darkgrid", color_codes=True)

#set colors for the strip plot
clrs = ['blue', 'red']

# plot a strip plot graph
ax = sns.swarmplot(x="embark_town", y="age", hue="alive", palette=clrs, data=df)

ax.legend(title = "Survive")

#show the strip plot graph
plt.show()


# In[21]:


# LAB 1 - Data Visualization
# Question 6 - Strip Plots

# Create an interactive strip plot where the data is grouped by the city the passengers embarked and by survival status

import plotly.express as px

fig = px.strip(df, x="embark_town", y="age", color="alive", labels={"alive": "Survive"}, stripmode="overlay")

fig.update_layout(xaxis_title="Passenger Class", yaxis_title="Age", title="Male passengers in first class over 18 years old" + "<br>" + "grouped by the city the passengers embarked and by survival status")

fig.show()


# In[ ]:




