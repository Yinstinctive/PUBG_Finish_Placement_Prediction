**The purpose of this exploratory data analysis is to get a basic understanding of the dataset and create a quick insight on the relations between some features.**
1. Read the data
    ```Python
    import pandas as pd
    import numpy as np
    import matplotlib.pyplot as plt
    import seaborn as sns
    
    file = r'C:\Users\yingk\Desktop\PUBG Data\train_V2.csv'
    train = pd.read_csv(file)
    pd.set_option('display.max_columns',30)
    ```
2. Overview
    ```Python
    train.columns
    train.shape
    train.head()
    train.info()
    ```
    There are 30 columns in the dataset.<br> 
    ![columns](https://github.com/Yinstinctive/PUBG_Finish_Placement_Prediction/blob/master/EDA_Images/columns.PNG)<br>
    The dataframe shape is: ![shape](https://github.com/Yinstinctive/PUBG_Finish_Placement_Prediction/blob/master/EDA_Images/shape.PNG)<br>
    Here is the head rows of the dataframe:<br>
    ![head](https://github.com/Yinstinctive/PUBG_Finish_Placement_Prediction/blob/master/EDA_Images/head.PNG)<br>
    ![info](https://github.com/Yinstinctive/PUBG_Finish_Placement_Prediction/blob/master/EDA_Images/info.PNG)<br>
    **Data fields**
    DBNOs - Number of enemy players knocked.<br>
    assists - Number of enemy players this player damaged that were killed by teammates.<br>
    boosts - Number of boost items used.<br>
    damageDealt - Total damage dealt. Note: Self inflicted damage is subtracted.<br>
    headshotKills - Number of enemy players killed with headshots.<br>
    heals - Number of healing items used.<br>
    Id - Player’s Id<br>
    killPlace - Ranking in match of number of enemy players killed.<br>
    killPoints - Kills-based external ranking of player. (Think of this as an Elo ranking where only kills matter.) If there is a value other than -1 in rankPoints, then any 0 in killPoints should be treated as a “None”.<br>
    killStreaks - Max number of enemy players killed in a short amount of time.<br>
    kills - Number of enemy players killed.<br>
    longestKill - Longest distance between player and player killed at time of death. This may be misleading, as downing a player and driving away may lead to a large longestKill stat.<br>
    matchDuration - Duration of match in seconds.<br>
    matchId - ID to identify match. There are no matches that are in both the training and testing set.<br>
    matchType - String identifying the game mode that the data comes from. The standard modes are “solo”, “duo”, “squad”, “solo-fpp”, “duo-fpp”, and “squad-fpp”; other modes are from events or custom matches.<br>
    rankPoints - Elo-like ranking of player. This ranking is inconsistent and is being deprecated in the API’s next version, so use with caution. Value of -1 takes place of “None”.<br>
    revives - Number of times this player revived teammates.<br>
    rideDistance - Total distance traveled in vehicles measured in meters.<br>
    roadKills - Number of kills while in a vehicle.<br>
    swimDistance - Total distance traveled by swimming measured in meters.<br>
    teamKills - Number of times this player killed a teammate.<br>
    vehicleDestroys - Number of vehicles destroyed.<br>
    walkDistance - Total distance traveled on foot measured in meters.<br>
    weaponsAcquired - Number of weapons picked up.<br>
    winPoints - Win-based external ranking of player. (Think of this as an Elo ranking where only winning matters.) If there is a value other than -1 in rankPoints, then any 0 in winPoints should be treated as a “None”.<br>
    groupId - ID to identify a group within a match. If the same group of players plays in different matches, they will have a different groupId each time.<br>
    numGroups - Number of groups we have data for in the match.<br>
    maxPlace - Worst placement we have data for in the match. This may not match with numGroups, as sometimes the data skips over placements.<br>
    winPlacePerc - The target of prediction. This is a percentile winning placement, where 1 corresponds to 1st place, and 0 corresponds to last place in the match. It is calculated off of maxPlace, not numGroups, so it is possible to have missing chunks in a match.<br>
3. Check Missing Values
    ```Python
    train.isna().sum()
    train[train['winPlacePerc'].isna()]
    train.dropna(inplace=True)
    ```
    ![missing values](https://github.com/Yinstinctive/PUBG_Finish_Placement_Prediction/blob/master/EDA_Images/missing%20value.PNG)<br>
4. Check Correlations
    ```Python
    corr = train.corr()
    plt.figure(figsize=(12,12))
    sns.heatmap(corr, cmap='coolwarm',linewidths=0.5)
    plt.show()
    ```
    ![corr_heatmap](https://github.com/Yinstinctive/PUBG_Finish_Placement_Prediction/blob/master/EDA_Images/corr_heatmap.png)<br>
    Proceed to further look into the features with high correlations.<br>
    ```Python
    high_related = corr.apply(lambda value:(np.abs(value)>0.8))
    plt.figure(figsize=(12,12))
    sns.heatmap(high_related,cbar=False, cmap='viridis',linewidths=0.5)
    ```
    ![high_related_heatmap](https://github.com/Yinstinctive/PUBG_Finish_Placement_Prediction/blob/master/EDA_Images/high_related_heatmap.png)<br>
    Proceed to check the high correlated features one by one.<br>
    ```Python
    #damageDealt and kills
    train.plot(x='damageDealt', y='kills', kind='scatter',figsize=(15,10))
    ```
    ![damageDealt and kills](https://github.com/Yinstinctive/PUBG_Finish_Placement_Prediction/blob/master/EDA_Images/damage%26kills_scatter.png)<br>
    ```Python
    #killPlace and kills
    train.plot(x='killPlace', y='kills', kind='scatter',figsize=(15,10))
    train[(train['killPlace']<10) & (train['kills']<=1)][['kills','killPlace']]
    ```
    ![killPlace and kills](https://github.com/Yinstinctive/PUBG_Finish_Placement_Prediction/blob/master/EDA_Images/killPlace%20and%20kills.png)<br>
    ![killPlace and kills](https://github.com/Yinstinctive/PUBG_Finish_Placement_Prediction/blob/master/EDA_Images/killPlace%20and%20kills2.png)<br>
    It seems there is some incosistency with killPlace and kills. There are players who kills 0 but have a high rank in killPlace. Need further investigate these two features.<br>
    ```Python
    #killPoints and rankPoints
    train.plot(x='killPoints', y='rankPoints', kind='scatter',figsize=(15,10))
    train[['killPoints','rankPoints']]
    train['killPoints'].value_counts()
    train['rankPoints'].value_counts()
    #killPoints and winPoints
    train.plot(x='killPoints', y='winPoints', kind='scatter',figsize=(15,10))
    train['winPoints'].value_counts()
    ```
    ![killPoints and rankPoints](https://github.com/Yinstinctive/PUBG_Finish_Placement_Prediction/blob/master/EDA_Images/killPoints%20and%20rankPoints.png)<br>
    ![killPoints and winPoints](https://github.com/Yinstinctive/PUBG_Finish_Placement_Prediction/blob/master/EDA_Images/killPoints%20and%20winPoints.png)<br>
    ![killPoints value counts](https://github.com/Yinstinctive/PUBG_Finish_Placement_Prediction/blob/master/EDA_Images/killPoints_value_counts.PNG)<br>
    ![rankPoints value counts](https://github.com/Yinstinctive/PUBG_Finish_Placement_Prediction/blob/master/EDA_Images/rankPoints_value_counts.PNG)<br>
    ![killPoints and winPoints](https://github.com/Yinstinctive/PUBG_Finish_Placement_Prediction/blob/master/EDA_Images/killPoints%20and%20winPoints.png)<br>
    ![winPoints value counts](https://github.com/Yinstinctive/PUBG_Finish_Placement_Prediction/blob/master/EDA_Images/winPoints_value_counts.PNG)<br>
    There are too many missing values in killPoints, rankPoints and winPoints. Therefore, not suggesting using these three features in prediction model.<br>
    ```Python
    #kills and killStreaks
    train.plot(x='kills', y='killStreaks', kind='scatter',figsize=(15,10))
    ```
    ![kills and killStreaks](https://github.com/Yinstinctive/PUBG_Finish_Placement_Prediction/blob/master/EDA_Images/kills%20and%20killStreaks.png)<br>
    ```Python
    #maxPlace and numGroups
    train.plot(x='maxPlace', y='numGroups', kind='scatter',figsize=(15,10))
    ```
    ![maxPlace and numGroups](https://github.com/Yinstinctive/PUBG_Finish_Placement_Prediction/blob/master/EDA_Images/maxPlace%20and%20numGroups.png)<br>
    maxPlace and numGroups are almost identical, according to data fields decription, suggest to drop maxPlace.<br>
    ```Python
    #walkDistance and winPlacePerc
    train.plot(x='walkDistance', y='winPlacePerc', kind='scatter',figsize=(15,10))
    train['walkDistance'].describe()
    def bin_walkDistance(walkDistance):
        if walkDistance<=155.1:
            return '0-25%'
        elif walkDistance<=685.6:
            return '25-50%'
        elif walkDistance<=1976:
            return '50-75%'
        else:
            return '75-100%'
    train['walkDistBins'] = train['walkDistance'].apply(bin_walkDistance)
    plt.figure(figsize=(15,10))
    sns.boxplot(x='walkDistBins',y='winPlacePerc',data=train,palette='rainbow',order=['0-25%','25-50%','50-75%','75-100%'])
    ```
    ![walkDistance and winPlacePerc_scatter](https://github.com/Yinstinctive/PUBG_Finish_Placement_Prediction/blob/master/EDA_Images/walkDistance%20and%20winPlacePerc_scatter.png)<br>
    ![walkDistance and winPlacePerc_box](https://github.com/Yinstinctive/PUBG_Finish_Placement_Prediction/blob/master/EDA_Images/walkDistance%20and%20winPlacePerc_box.png)<br>
    
