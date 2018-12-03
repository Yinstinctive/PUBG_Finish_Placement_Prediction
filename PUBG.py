import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

data = pd.read_csv(r'C:\Users\yingk\Desktop\PUBG Data\train_V2.csv')
# drop Id, groupId, matchId, killPoints, rankPoints, winPoints, maxPlace,winPlacePerc(target)
features = data.drop(['Id','groupId','matchId','killPoints','rankPoints','winPoints','maxPlace'],axis=1)
features.dropna(inplace=True)
target = features['winPlacePerc']
features.drop(['winPlacePerc'],axis=1,inplace=True)


#check missing values with heatmap
#sns.heatmap(features.isnull(),yticklabels = False, cbar=False, cmap='viridis')
#plt.show()

# matchType need to be separated, fpp or not; solo/duo/squad could be indicated with numGroups
def is_fpp(matchType):
    if matchType[-3:]=='fpp':
        return 1
    else:
        return 0
features['fpp'] = features['matchType'].apply(is_fpp)

def categorize_type(matchType):
    s = matchType[:3]
    if s == 'sol':
        return 'solo'
    elif s=='duo':
        return 'duo'
    elif s=='squ':
        return 'squad'
    elif s=='cra':
        return 'crash'
    elif s=='fla':
        return 'flare'
    elif s=='nor':
        return f"normal-{matchType.split('-')[1]}"
features['matchType'] = features['matchType'].apply(categorize_type)

#Convert Categorical Features to Dummy Variable
matchType = pd.get_dummies(features['matchType'],drop_first=True)
features.drop(['matchType'],axis=1,inplace=True)
features = pd.concat([features,matchType],axis=1)

#Train Test Split
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(features, target, test_size=0.3, random_state=101)

#Standardize the Variables(for KNN and PCA)
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
scaler.fit(X_train)
scaled_X_train = scaler.transform(X_train)
df_scaled_X_train = pd.DataFrame(scaled_X_train,columns=X_train.columns)

scaler_test = StandardScaler()
scaler_test.fit(X_test)
scaled_X_test = scaler_test.transform(X_test)
df_scaled_X_test = pd.DataFrame(scaled_X_test, columns=X_test.columns) 

#Check NaN
print(f"scaled X_train have NaN? {np.any(np.isnan(df_scaled_X_train))}")
print(f"scaled df_scaled_X_train are all finite? {np.all(np.isfinite(df_scaled_X_train))}")   
print(f"y_train has NaN? {np.any(np.isnan(y_train))}")
print(f"y_train is all finite? {np.all(np.isfinite(y_train))}")

#Linear Regression
from sklearn.linear_model import LinearRegression
lm=LinearRegression()
lm.fit(X_train, y_train)
lm_pred = lm.predict(X_test)

#Evaluate Linear Regression
from sklearn.metrics import mean_absolute_error
lm_MAE = mean_absolute_error(y_test, lm_pred)
print('MAE for linear regression is:', lm_MAE)

#Principal Component Analysis
MAE_PCA = []
from sklearn.decomposition import PCA
for components_num in range(10,29):
    pca = PCA(n_components=components_num)
    pca.fit(df_scaled_X_train)
    X_train_pca = pca.transform(df_scaled_X_train)

#Plot the two dimensions
#plt.figure(figsize=(8,6))
#plt.scatter(X_train_pca[:,0],X_train_pca[:,1],c=y_train,cmap='plasma')
#plt.xlabel('First principal component')
#plt.ylabel('Second Principal Component')
#plt.show()

    pca_test = PCA(n_components=components_num)
    pca_test.fit(df_scaled_X_test)
    X_test_pca = pca.transform(df_scaled_X_test)

    #Use PCA components to fit linear regression
    pca_lm=LinearRegression()
    pca_lm.fit(X_train_pca,y_train)
    pca_lm_pred = pca_lm.predict(X_test_pca)

    #Evaluate linear regression with principle components
    pca_lm_MAE = mean_absolute_error(y_test, pca_lm_pred)
    print('MAE for linear regression with principle components is:', pca_lm_MAE)
    MAE_PCA.append(pca_lm_MAE)


