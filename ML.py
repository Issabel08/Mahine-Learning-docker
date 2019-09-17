# -*- coding: utf-8 -*-

#%%
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib
import matplotlib.pyplot as plt
from scipy import stats
from sklearn.ensemble import RandomForestClassifier
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import RFECV
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import cross_val_score
from sklearn import preprocessing

matplotlib.rcParams['figure.figsize'] = [10,10]
matplotlib.rcParams['font.size'] = 20

import warnings
warnings.filterwarnings("ignore")

np.random.seed(42)

#%%
def outliers_col(df):
    print('Number of ouliers with z_core>3 :')
    outliers_list=list()
    for columna in df:
        if df[columna].dtype != np.object:
            n_outliers = len(df[np.abs(stats.zscore(df[columna])) > 3])    
            print("{} | {} | {}".format(
                df[columna].name,
                n_outliers,
                df[columna].dtype
        ))
            if n_outliers>0:
                outliers_list.append(columna) 
    print('Columns with outliers: ')
    return outliers_list

def missing_values(df):
    print('Number of missing values for feature:')
    print('From {} values'.format(len(df)))
    drop_features=list()
    cont=0
    for col in df:
        nu_mv=len(df)-df[col].notnull().sum()
        if nu_mv>=1:
            per_missing=(nu_mv*100)/len(df)
            print('{}:  {} missing values {:.2f}% of feature data'.format( col,nu_mv,per_missing))
            if per_missing>40.0:
                drop_features.append(col)
            cont +=nu_mv
    if cont==0:
            print('...No missing values')
    
    for col in drop_features:
        df=df.drop([col],axis=1)
        print('\nDrop column {} with missing values > 40%'.format(col))
    return df

def important_features(df,y_all):
    from sklearn.feature_selection import RFE
    clf = RandomForestClassifier(random_state=42)
    selector = RFE(clf, 1, step=1)
    selector = selector.fit(df, y_all)
    pos=selector.ranking_
    return pos

def select_features(no_feat,df,y):
    pos=important_features(df,y)
    no_features=no_feat
    df_pos=pd.DataFrame({'Columna':df.columns,'Pos':pos})
    for n in range(1,df.shape[1]+1):  
        no_feat=n
        i=1
        new_data=[]
        while i<=no_features:
            for j in range(len(df_pos)):
                if df_pos.values[j,1]==i:
                    agrega=df_pos.values[j,0]
                    new_data.append(agrega)
                    break
            i=i+1
    print('Set of selected features : ',new_data)
    return sorted(new_data)

def model_selection(df,y):
    clf = RandomForestClassifier(random_state=42)
    model=cross_val_score(clf,df,y,cv=10)
    return model

def results_all_datas(df,y):
    results=[]
    for no_features in range(1,df.shape[1]+1):
        cols=select_features(no_features,df,y)
        df_new=df[cols]
        r=model_selection(df_new,y)
        results.append(r.mean())
    return results
#%%
#https://archive.ics.uci.edu/ml/datasets/Las+Vegas+Strip
print('---------- Las Vegas Strip Data Set  ----------')
print('\n')
print('Abstract: This dataset includes quantitative and categorical features from online reviews from 21 hotels located in Las Vegas Strip, extracted from TripAdvisor ')
print('The dataset contains 504 records and 20 tuned features, 24 per hotel (two per each month, randomly selected), regarding the year of 2015.')

print('Download data: https://archive.ics.uci.edu/ml/datasets/Las+Vegas+Strip ')

print('\n')
print('Reading data to training...')    
df=pd.read_csv('data/data.csv', sep=';') 
       
print('\n')
print('Data: ')
print(df.head())
#%%
print('\n')
print('Size of data: ')
print(df.shape)
#%%
y=df['Score']
df=df.drop('Score', axis=1)
#%%
df.dtypes
#%%
print('\n')
print('Missing valu?? ')
df=missing_values(df)

#%%
print('\n')
print('Drop rows duplicated!! ')
df = df.drop_duplicates()
#%%
print('\n')
print('Drop columns duplicated!! ')
df=df.T.drop_duplicates().T

#%%
print('\n')
print('Selecting numeric by data_type..')
numeric_columns = df.select_dtypes([np.number]).columns
categorical_columns = df.select_dtypes([object]).columns
#%%
print('\n')
print('Encoder data..')
encoder=preprocessing.LabelEncoder()
for col in df:
    if df[col].dtype == np.object:
        df[col]=encoder.fit_transform(df[col] )
#%%
print('\n')
print('Data Description')
print(df.describe(include='all'))
#%%
print('\n')
print('Tabla of correlations: ')
corr=df.corr()
print(corr)
#%%
#print('\n')
#print('Heatmap: ')
sns.heatmap(corr)
#%%
#print('\n')
#print('Boxplot: ')
df_boxplot=df.boxplot(grid=False, rot=45, fontsize=8)
#%%
print('\n')
print('Outliers: ')
print(outliers_col(df))

#%%
list_outliers=outliers_col(df)
#%%
print('\n')
print('Data description of features with outliers ')
for col in list_outliers:
    print('\n')
    print(df[col].describe())
#%%
plt.scatter(df['Nr. reviews'], y, alpha=0.5)
plt.title('Scatter plot')
plt.xlabel('Nr. reviews')
plt.ylabel('Score')
plt.show()
#%%
plt.scatter(df['Nr. hotel reviews'], y, alpha=0.5)
plt.title('Scatter plot')
plt.xlabel('Nr. hotel reviews')
plt.ylabel('Score')
plt.show()
#%%
plt.scatter(df['Helpful votes'], y, alpha=0.5)
plt.title('Scatter plot')
plt.xlabel('Scatter plot')
plt.ylabel('Score')
plt.show()
#%%
print('\n')
print('Preprocessing data.. ')
scaler=StandardScaler()
df_scaled=scaler.fit_transform(df)
df_scaled=pd.DataFrame(data=df_scaled, columns=df.columns)

#%%
print('\n')
print('Feature Selection with RFECV ')
estimator = DecisionTreeClassifier(random_state=42)
selector = RFECV(estimator, step=10, cv=100)
selector = selector.fit(df_scaled,y)
#%%
df_new=selector.transform(df_scaled)

#%%
no_features=df_new.shape[1]
#%%
print('\n')
print('Training the model using GridSearchCV with differents PCA() dimensions ')
#%%
list_rmse=list()
for no_feat in range(1,no_features+1):
    #%%
    svd = PCA(n_components=no_feat)
    df_truncated=svd.fit_transform(df_new)   
#%%#   
    X_train, X_test, y_train, y_test = train_test_split(df_truncated, y, test_size=0.3, random_state=42)
    
    #%%
    parameters = {'kernel':('linear', 'rbf'), 'C':[1, 10]}
    svc = SVC(gamma="scale")
    clf = GridSearchCV(svc, parameters, cv=int(np.round(X_train.shape[0]/10)))
    clf.fit(X_train, y_train)
 
    #%%
    predictions=clf.predict(X_test)
    #%%
    from sklearn.metrics import mean_squared_error
    mse=mean_squared_error(y_test, predictions)
    list_rmse.append(np.sqrt(mse))

#%%
range_=range(1,no_features+1)
#%%
print('\n')
print('Results:')
r=pd.DataFrame(np.linspace(1, no_features, no_features,dtype=int),columns=['No_features'])
r['RMSE error']=list_rmse
print(r)
#%%

comparison_results=pd.DataFrame(y_test)
comparison_results['predictions']=predictions
#%%
comparison_results.to_csv ('results/results.csv', index = None, header=True)
#%%

#%%
line=plt.plot(range_,list_rmse,linewidth=2)
plt.title('RMSE error')
plt.xlabel('Number of features with PCA')
plt.ylabel('RMSE')
plt.xlim([1,no_features])
plt.show()

