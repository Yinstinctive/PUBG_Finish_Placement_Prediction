import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error

train = pd.read_csv(r'C:\Users\yingk\Desktop\PUBG Data\train_V2.csv')

#1. Exploratory Data Analysis and Data Preprocessing

#Check correlations
train.info()
corr = train.corr()
high_related = corr.apply(lambda value:np.abs(value)>0.7)
plt.figure(figsize=(12,12))
sns.heatmap(high_related,cbar=False, cmap='viridis',linewidths=0.5)
plt.show()
#damageDealt&kills, killPlace&kills, killPoints&rankPoints, killPoints&winPoints, kills&killStreaks, killStreaks&killPlace,maxPlace&numGroups, 
#rankPoints&killPoints, rankPoints&winPoints,winPoints&killPoints
#walkDistance&winPlacePerc
train_screened = train.drop(['damageDealt','killPlace','killPoints','winPoints','killStreaks','killPlace','maxPlace'],axis=1)

#walkDistance&winPlacePerc
plt.figure(figsize=(20,6))
sns.distplot(train_screened['walkDistance'],kde=False, bins=50)
train_screened['walkDistance'].describe()
#685.6 50%

train_screened['lowWalkDistance'] = train_screened['walkDistance'].apply(lambda distance:distance<=685.6)
sns.boxplot(x='lowWalkDistance',y='winPlacePerc',data=train_screened,palette='coolwarm')

train_screened.drop(['walkDistance'],axis=1,inplace=True)


#Check missing values
plt.figure(figsize=(12,24))
sns.heatmap(train_screened.isnull(), cmap='viridis',cbar=False, yticklabels=False)

train_screened.dropna(inplace=True)

train_screened.drop(['Id','groupId','matchId'],axis=1,inplace=True)

#matchType
matchType = pd.get_dummies(train_screened['matchType'],drop_first=True)
train_screened.drop(['matchType'],axis=1,inplace=True)
train_screened = pd.concat([train_screened,matchType],axis=1)

#2. Modeling

#Train Test Split
data_lowWalk = train_screened[train_screened['lowWalkDistance']==True]
data_lowWalk.drop(['lowWalkDistance'],axis=1,inplace=True)
data_highWalk = train_screened[train_screened['lowWalkDistance']==False]
data_highWalk.drop(['lowWalkDistance'],axis=1,inplace=True)

X1_train, X1_test, y1_train, y1_test = train_test_split(data_lowWalk.drop(['winPlacePerc'],axis=1), data_lowWalk['winPlacePerc'], test_size=0.3)
X2_train, X2_test, y2_train, y2_test = train_test_split(data_highWalk.drop(['winPlacePerc'],axis=1), data_highWalk['winPlacePerc'], test_size=0.3)

#Train model
lm_lowWalk = LinearRegression()
lm_highWalk = LinearRegression()

lm_lowWalk.fit(X1_train,y1_train)
lm_highWalk.fit(X2_train,y2_train)

#Evaluate model
lowWalk_pred = lm_lowWalk.predict(X1_test)
highWalk_pred = lm_highWalk.predict(X2_test)

MAE_lowWalk = mean_absolute_error(y1_test,lowWalk_pred)
MAE_highWalk = mean_absolute_error(y2_test,highWalk_pred)

print('MAE_lowWalk: ',MAE_lowWalk)
print('MAE_highWalk: ',MAE_highWalk)