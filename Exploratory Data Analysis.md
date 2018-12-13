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
    ![columns](https://drive.google.com/open?id=1P_UAde100RPKbVm_8qI14QG0ntW0wi_h)<br>
    ![shape](https://drive.google.com/open?id=1aecKJcm3YJN925r_E3SMaUZy04W-vDVa)<br>
    ![head](https://drive.google.com/open?id=1Ur7PE5ngWj1Aso8Gfjx9OdJM18NklfHq)<br>
    ![info](https://drive.google.com/open?id=1dWdJzdrHg1T2slXPnJM5Gb9t-EtdYNoZ)<br>
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
    ![missing values](https://drive.google.com/open?id=1-B4ahT2Ng5uQZBa8V77vArrYXbKWhExv)<br>
4. Check Correlations
    ```Python
    corr = train.corr()
    plt.figure(figsize=(12,12))
    sns.heatmap(corr, cmap='coolwarm',linewidths=0.5)
    plt.show()
    ```
    ![corr_heatmap](https://drive.google.com/open?id=17a4Ix2P4bfPYBqgm61pl_AtO_1YZE7LH)<br>
    Proceed to further look into the features with high correlations.<br>
    ```Python
    high_related = corr.apply(lambda value:(np.abs(value)>0.8))
    plt.figure(figsize=(12,12))
    sns.heatmap(high_related,cbar=False, cmap='viridis',linewidths=0.5)
    ```
    ![high_related_heatmap](https://drive.google.com/open?id=1ii4bgtHupZx-1-slM2fzfayGlFS0autG)
    Proceed to check the high correlated features one by one.<br>
    ```Python
    #damageDealt and kills
    train.plot(x='damageDealt', y='kills', kind='scatter',figsize=(15,10))
    ```
    ![damageDealt and kills](https://drive.google.com/open?id=1FVmkIo6hlFdvVjMJoS_e4fJgYqJms3dd)<br>
    ```Python
    #killPlace and kills
    train.plot(x='killPlace', y='kills', kind='scatter',figsize=(15,10))
    train[(train['killPlace']<10) & (train['kills']<=1)][['kills','killPlace']]
    ```
    ![killPlace and kills](https://drive.google.com/open?id=1fA1V9BRxGFiXiHqhvWuGwHnT7COGDzfR)<br>
    ![killPlace and kills](https://drive.google.com/open?id=17bd76cqNCoMlBO2cgdMDqa177kwCFE_z)<br>
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
    ![killPoints and rankPoints](https://drive.google.com/open?id=1atQpb5Dq4WCKnE4I8cuArIjFnca6hmOY)<br>
    ![killPoints and winPoints](https://drive.google.com/open?id=1GWx4d91XIbNmKVNyRl2hFRYE2zLZTlRD)<br>
    ![killPoints value counts](https://drive.google.com/open?id=1GdBi9OdkLBiUusTFp3HwDa-_zsIuFKV2)<br>
    ![rankPoints value counts](https://drive.google.com/open?id=1XRcHJwAMY5ycTZA3-W1Q_cStEpnKCalk)<br>
    ![winPoints value counts](https://drive.google.com/open?id=1FaWEPGTvhdPJ9QW63WiyWun3muzAg3NP)<br>
    There are too many missing values in killPoints, rankPoints and winPoints. Therefore, not suggesting using these three features in prediction model.<br>
    ```Python
    #kills and killStreaks
    train.plot(x='kills', y='killStreaks', kind='scatter',figsize=(15,10))
    ```
    ![kills and killStreaks](https://drive.google.com/open?id=13E0bnPJleq1ZtjYcY901pwbUbnEfMiBg)<br>
    ```Python
    #maxPlace and numGroups
    train.plot(x='maxPlace', y='numGroups', kind='scatter',figsize=(15,10))
    ```
    ![maxPlace and numGroups](https://drive.google.com/open?id=1awz4o_n4PsDJXiVRs6jOpCc1VmwtQC3v)<br>
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
    ![walkDistance and winPlacePerc_scatter](https://drive.google.com/open?id=1Z7sione-utvI6jA8ynyJqvtokat_gpYi)<br>
    ![walkDistance and winPlacePerc_box](https://drive.google.com/open?id=1awt0neA52p0QRZIh6qh4YouiDS_7dVIp)<br>
    
