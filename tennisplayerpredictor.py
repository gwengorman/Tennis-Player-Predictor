#!/usr/bin/env python
# coding: utf-8

# # Tennis Player Predictor
# 
# This Linear regression model predictis tennis player performance based on their playing habits. Using **tennis_stats.csv** we will be analyzing and modeling data from the Association of Tennis Professionals (ATP), this model identifies the key factors that contribute to becoming one of the world's top tennis players.
# 
# Data from the top 1500 ranked players in the ATP over the span of 2009 to 2017 are provided in **tennis_stats.csv**:
# 
# Identifying Data
# - `Player` : name of the tennis player
# - `Year` : year data was recorded
#   
# Service Game Columns (Offensive)
# - `Aces` : number of serves by the player where the receiver does not touch the ball
# - `DoubleFaults` : number of times player missed both first and second serve attempts
# - `FirstServe` : % of first-serve attempts made
# - `FirstServePointsWon` : % of first-serve attempt points won by the player
# - `SecondServePointsWon`: % of second-serve attempt points won by the player
# - `BreakPointsFaced`: number of times where the receiver could have won service game of the player
# - `BreakPointsSaved`: % of the time the player was able to stop the receiver from winning service game when they had the chance
# - `ServiceGamesPlayed`: total number of games where the player served
# - `ServiceGamesWon`: total number of games where the player served and won
# - `TotalServicePointsWon`: % of points in games where the player served that they won
#   
# Return Game Columns (Defensive)
# - `FirstServeReturnPointsWon`: % of opponents first-serve points the player was able to win
# - `SecondServeReturnPointsWon`: % of opponents second-serve points the player was able to win
# - `BreakPointsOpportunities`: number of times where the player could have won the service game of the opponent
# - `BreakPointsConverted`: % of the time the player was able to win their opponent’s service game when they had the chance
# - `ReturnGamesPlayed`: total number of games where the player’s opponent served
# - `ReturnGamesWon`: total number of games where the player’s opponent served and the player won
# - `ReturnPointsWon`: total number of points where the player’s opponent served and the player won
# - `TotalPointsWon`: % of points won by the player
# 
# Outcomes
# - `Wins`: number of matches won in a year
# - `Losses`: number of matches lost in a year
# - `Winnings`: total winnings in USD($) in a year
# - `Ranking`: ranking at the end of year

# In[1]:


import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from scipy.stats import pearsonr

# load and investigate the data:
df = pd.read_csv('tennis_stats.csv')
df.head()


# In[3]:


offensive = ["Aces", "DoubleFaults", "FirstServe", "FirstServePointsWon", "SecondServePointsWon", "BreakPointsFaced", "BreakPointsSaved", "ServiceGamesPlayed", "ServiceGamesWon", "TotalServicePointsWon"]
defensive = ["FirstServeReturnPointsWon", "SecondServeReturnPointsWon","BreakPointsOpportunities", "BreakPointsConverted", "ReturnGamesPlayed", "ReturnGamesWon", "ReturnPointsWon", "TotalPointsWon"]
outcomes = ["Winnings"]


# In[5]:


df.info()


# There are 1721 observations, 23 columns and no `null` values. The data types align with each variable so we are OK to begin exploratory analysis.

# In[8]:


# perform exploratory analysis here:
df.describe(include = 'all')


# It looks like there may be an outlier for the max value of `Aces` but we will perform further analysis to explore

# In[11]:


#checking to make sure there are no null values
df[df.isnull().any(axis=1)]


# In[13]:


columns = df.columns.tolist()
print(columns)
for column in columns:
    if column != 'Player':  # Skip the 'Player' column
        correlation, p = pearsonr(df[column], df['Winnings'])
        print(f"Correlation between {column} and Winnings: {correlation}")
    plt.scatter(x=column, y= 'Winnings', data=df, alpha=0.5)
    plt.xlabel(column)
    plt.ylabel('Winnings $')
    plt.show()
    plt.clf()
    


# In[18]:


#heatmap to test for multicollinearity
numeric_df = df[offensive+outcomes] #too many predictor variables, so we seperated
corr_matrix = numeric_df.corr()
# Create the heatmap
colors = sns.diverging_palette(150, 275, s=80, l=55, n=9, as_cmap=True)
sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', center=0, robust=True)
# Show the heatmap
plt.show()
plt.clf()


# In[20]:


#heatmap to test for multicollinearity
numeric_df = df[defensive+outcomes] #too many predictor variables, so we seperated
# Create the heatmap
corr_matrix = numeric_df.corr()
colors = sns.diverging_palette(150, 275, s=80, l=55, n=9, as_cmap=True)
sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', center=0, robust=True)
# Show the heatmap
plt.show()
plt.clf()


# We aknowledge the presence of multicollinearity among certain variables, and would need to remove these variables from any regression model. However for this excersice, we are focused on testing the correlation of independent variables to the target variable, `Winnings`. From these two heatmaps we see high correlation between `Aces`, `DoubleFaults`, `BreakPointsFaced`,`ServiceGamesPlayed`,`BreakPointsOpportunities`, and `ReturnGamesPlayed`.

# In[23]:


##single feature linear regressions

aces = df[['Aces']]
winnings = df[['Winnings']]

aces_train, aces_test, winnings_train, winnings_test = train_test_split(aces, winnings, train_size = 0.8)

model = LinearRegression()
model.fit(aces_train,winnings_train)

# score model on test data
print('Predicting Winnings with Aces Test Score:', model.score(aces_test,winnings_test))

# make predictions with model
winnings_prediction = model.predict(aces_test)

# plot predictions against actual winnings
plt.scatter(winnings_test,winnings_prediction, alpha=0.4)
plt.title('Predicted Winnings vs. Actual Winnings - 1 Feature')
plt.xlabel('Actual Winnings')
plt.ylabel('Predicted Winnings')
plt.show()
plt.clf()


# In[25]:


## perform two feature linear regressions here:

twofeature = df[['Aces','ReturnGamesPlayed']]
winnings = df[['Winnings']]

twofeature_train, twofeature_test, winnings_train, winnings_test = train_test_split(twofeature, winnings, train_size = 0.8)

model = LinearRegression()
model.fit(twofeature_train,winnings_train)

# score model on test data
print('Predicting Winnings Test Score:', model.score(twofeature_test,winnings_test))

# make predictions with model
winnings_prediction = model.predict(twofeature_test)

# plot predictions against actual winnings
plt.scatter(winnings_test,winnings_prediction, alpha=0.4)
plt.title('Predicted Winnings vs. Actual Winnings - 2 Feature')
plt.xlabel('Actual Winnings')
plt.ylabel('Predicted Winnings')
plt.show()
plt.clf()


# Just by adding one additional variable with high correlation we see the models accuracy score go from 58% to 82%.

# In[36]:


## multivariate linear regressions
multifeature = df[['Aces','ReturnGamesPlayed','Aces', 'DoubleFaults', 
                 'BreakPointsFaced','ServiceGamesPlayed','BreakPointsOpportunities', 'ReturnGamesPlayed']]
winnings = df[['Winnings']]

multifeature_train, multifeature_test, winnings_train, winnings_test = train_test_split(multifeature, winnings, train_size = 0.8)

model = LinearRegression()
model.fit(multifeature_train,winnings_train)

# score model on test data
print('Predicting Winnings Test Score:', model.score(multifeature_test,winnings_test))

# make predictions with model
winnings_prediction = model.predict(multifeature_test)

# plot predictions against actual winnings
plt.scatter(winnings_test,winnings_prediction, alpha=0.4)
plt.title('Predicted Winnings vs. Actual Winnings - Multivariate')
plt.xlabel('Actual Winnings')
plt.ylabel('Predicted Winnings')
plt.show()
plt.clf()


# Our model can now predict winnings with a test score of 86%.

# In[ ]:




