#!/usr/bin/env python
# coding: utf-8

# In[1]:


# LAB 5 - Supervised Learning
# Question 1 - Logistic Regression using logit

# import the necessary libraries
import pandas as pd
import numpy as np
import statsmodels.formula.api as sms
from sklearn.model_selection import train_test_split

# load nbaallelo.csv into a dataframe
# code to load csv file
df = pd.read_csv("nbaallelo.csv")

# Converts the feature "game_result" to a binary feature and adds as new column "wins"
wins = df.game_result == "W"
bool_val = np.multiply(wins, 1)
wins = pd.DataFrame(bool_val, columns = ["game_result"])
wins_new = wins.rename(columns = {"game_result": "wins"})
df_final = pd.concat([df, wins_new], axis=1) 

# split the data df_final into training and test sets with a test size of 0.3 and random_state = 0
# code to split df_final into training and test sets
train, test = train_test_split(df_final, test_size=0.3, random_state=0)

# construct a logistic model with wins as the target and pts as the predictor, using the training set
# code to construct logistic model using the logit function
lm = sms.logit("wins~pts", data=train).fit()

# print coefficients for the model
# code to return coefficients
print(lm.params)

print("\n")

# construct a logistic model with wins as the target and elo_i as the predictor, using the training set
# code to construct logistic model using the logit function
lm = sms.logit("wins~elo_i", data=train).fit()

# print coefficients for the model
# code to return coefficients
print(lm.params)


# In[2]:


# LAB 5 - Supervised Learning
# Question 2 - Logistic Regression using Logistic Regression

# import the necessary libraries
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix

# load nbaallelo.csv into a dataframe
# code to load csv file
df = pd.read_csv("nbaallelo.csv")

# Converts the feature "game_result" to a binary feature and adds as new column "wins"
wins = df.game_result == "W"
bool_val = np.multiply(wins, 1)
wins = pd.DataFrame(bool_val, columns = ["game_result"])
wins_new = wins.rename(columns = {"game_result": "wins"})
df_final = pd.concat([df, wins_new], axis=1) 

# split the data df_final into training and test sets with a test size of 0.3 and random_state = 0
# code to split df_final into training and test sets
train, test = train_test_split(df_final, test_size=0.3, random_state=0)

# build the logistic model using the LogisticRegression function with wins as the target variable and pts as the predictor.
model = LogisticRegression()
model.fit(train[["pts"]], train["wins"])

# use the test set to predict the wins from the pts
# code to predict wins
predictions = model.predict(test[["pts"]])

# generate confusion matrix
# code to generate confusion matrix
conf = confusion_matrix(test['wins'], predictions)

print("confusion matrix based on pts is \n", conf)

# calculate the sensitivity
# code to calculate the sensitivity
sens = conf[0,0]/(conf[0,0]+conf[0,1])
print("Sensitivity is ", sens)

# calculate the specificity
# code to calculate the specificity
spec = conf[1,1]/(conf[1,0]+conf[1,1])
print ("Specificity is ", spec)

print("\n")

# build the logistic model using the LogisticRegression function with wins as the target variable and elo_i as the predictor.
model = LogisticRegression()
model.fit(train[["elo_i"]], train["wins"])

# use the test set to predict the wins from the elo_i score
# code to predict wins
predictions = model.predict(test[["elo_i"]])

# generate confusion matrix
# code to generate confusion matrix
conf = confusion_matrix(test['wins'], predictions)

print("confusion matrix based on elo_i score is \n", conf)

# calculate the sensitivity
# code to calculate the sensitivity
sens = conf[0,0]/(conf[0,0]+conf[0,1])
print("Sensitivity is ", sens)

# calculate the specificity
# code to calculate the specificity
spec = conf[1,1]/(conf[1,0]+conf[1,1])
print ("Specificity is ", spec)


# In[ ]:




