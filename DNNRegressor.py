# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""
import tensorflow as tf
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error

#drop features, return df_feature and df_target
def get_feature_target(file_name):
    data = pd.read_csv(file_name)
    # drop Id, groupId, matchId, killPoints, rankPoints, winPoints, maxPlace,winPlacePerc(target)
    features = data.drop(['Id','groupId','matchId','killPoints','rankPoints','winPoints','maxPlace'],axis=1)
    features.dropna(inplace=True)
    target = features['winPlacePerc']
    features.drop(['winPlacePerc'],axis=1,inplace=True)
    return [features, target]

# matchType need to be separated, fpp or not; solo/duo/squad could be indicated with numGroups
def is_fpp(matchType):
    if matchType[-3:]=='fpp':
        return 1
    else:
        return 0

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

def pre_processing(features):
    features['fpp'] = features['matchType'].apply(is_fpp)
    features['matchType'] = features['matchType'].apply(categorize_type)
    
    #Convert Categorical Features to Dummy Variable
    matchType = pd.get_dummies(features['matchType'],drop_first=True)
    features.drop(['matchType'],axis=1,inplace=True)
    features = pd.concat([features,matchType],axis=1)
    return features

def DNN_modeling(file_name):
    feat_tar = get_feature_target(file_name)
    feat = feat_tar[0]
    target = feat_tar[1]
    features = pre_processing(feat)
    
    #Train Test Split
    X_train, X_test, y_train, y_test = train_test_split(features, target, test_size=0.3, random_state=101)
    
    #DNN
    feat_cols = []
    for col in features:
        feat_cols.append(tf.feature_column.numeric_column(col))
    input_func = tf.estimator.inputs.pandas_input_fn(x=X_train,y=y_train,shuffle=True,batch_size=20)
    classifier = tf.estimator.DNNRegressor(hidden_units=[10,20,10],feature_columns=feat_cols)
    classifier.train(input_fn=input_func,steps=50)
    
    #Model Evaluation
    pred_fn = tf.estimator.inputs.pandas_input_fn(x=X_test,batch_size=len(X_test),shuffle=False)
    predictions = list(classifier.predict(input_fn=pred_fn))
    final_pred = []
    for pred in predictions:
        final_pred.append(pred['predictions'][0])
    print(mean_absolute_error(y_test,final_pred))
    return classifier

train = r'C:\Users\yingk\Desktop\PUBG Data\train_V2.csv'    
dnn = DNN_modeling(train)
      
test = r'C:\Users\yingk\Desktop\PUBG Data\test_V2.csv'

def get_test_features(file_name):
    data = pd.read_csv(file_name)
    # drop Id, groupId, matchId, killPoints, rankPoints, winPoints, maxPlace,winPlacePerc(target)
    features = data.drop(['Id','groupId','matchId','killPoints','rankPoints','winPoints','maxPlace'],axis=1)
    features = pre_processing(features)
    return features

processed_test_features = get_test_features(test)
dnn_predictions = linear_regression.predict(processed_test_features)
Id = pd.read_csv(test)['Id'].to_frame()
lm_predictions = pd.Series(lm_predictions).to_frame()
lm_predictions.columns = ['winPlacePerc']
result = Id.join(lm_predictions)
result.to_csv('prediction.csv',index=False)