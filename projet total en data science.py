# -*- coding: utf-8 -*-
"""
Created on Mon Apr 15 11:27:23 2019

@author: HP User
"""

import pandas as pd
import numpy as np
from sklearn import preprocessing
from sklearn.preprocessing import LabelEncoder,OneHotEncoder
from sklearn.preprocessing import StandardScaler
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib as mpl


#import data of the file
data=pd.read_csv('C:/Users/HP User/Desktop/mes etudes/formation bigdata/machine learning/KAG_conversion_data.csv')
#take a quick look at the data,the size and the columns of the file
print(data.columns)
print(data.shape)
print data.iloc[:]


#create a copy of data for editing
dataTF=data.copy()
#abreviate some variable names
dataTF.rename(columns={'xyz_campaign_id': 'xyzCampId','fb_campaign_id': 'fbCampId','Impressions': 'impr','Total_Conversion': 'conv','Approved_Conversion': 'appConv'}, inplace=True)
dataTF = dataTF.drop(dataTF[dataTF.xyzCampId != 1178].index)
#show the results
dataTF.columns
#show all dataframe
print dataTF

#look for unique values in 'age' column
dataTF['age'].unique()
#look for unique values in 'gender' column
dataTF['gender'].unique()


X=dataTF[['age','gender']]
print X
X=pd.get_dummies(data=X)

dataTF=pd.concat([dataTF,X],axis=1)
dataTF=dataTF.drop(['age','gender'],axis=1)
print dataTF.iloc[:]
dataTF.shape
 

#creating the additionnal features
dataTF['CTR']=(dataTF['Clicks']/dataTF['impr'])*100
dataTF['CTR']=dataTF['CTR'].astype('float')
dataTF['CPC']=(dataTF['Spent']/dataTF['Clicks'])
dataTF['CPC']=dataTF['CPC'].astype('float')

dataTF['totConv']=dataTF['conv']+dataTF['appConv']
dataTF['totConv']=dataTF['totConv'].astype('int')



dataTF['appConVal']=dataTF['appConv']*100
dataTF['appConVal']=dataTF['appConVal'].astype('int')

dataTF['conVal']=dataTF['conv']*5
dataTF['conVal']=dataTF['conVal'].astype('int')


dataTF['totConVal']=dataTF['conv']*5+dataTF['appConv']*100
dataTF['totConVal']=dataTF['totConVal'].astype('int')

dataTF['CPM']=(dataTF['Spent']/dataTF['impr'])*1000
dataTF['CPM']=dataTF['CPM'].astype('float')

dataTF['costPerCon']=(dataTF['Spent']/dataTF['totConv'])
dataTF['costPerCon']=dataTF['costPerCon'].astype('float')


dataTF['ROAS']=(dataTF['totConVal']/dataTF['Spent'])
dataTF['ROAS']=dataTF['ROAS'].astype('float')


#compare between data and dataTF
print(dataTF.iloc[:])
print (data.iloc[:])

encoded=dataTF[['ad_id','xyzCampId', 'fbCampId', 'interest', 'impr', 'Clicks', 'Spent',
       'conv', 'appConv', 'totConv', 'appConVal', 'conVal', 'totConVal', 'CPM','costPerCon',
       'ROAS', 'gender_F', 'gender_M', 'age_30-34', 'age_35-39',
       'age_40-44', 'age_45-49']].apply(LabelEncoder().fit_transform)

bigdata =pd.concat([encoded], axis=0)
                     
bigdata
datacat=bigdata.copy()
datacat.to_csv('C:/Users/HP User/Desktop/datacat.csv',index=False) 


dataTF.dtypes
bigdata.dtypes


df=pd.read_csv('C:/Users/HP User/Desktop/datacat.csv')

df.head()
df.info()
df.columns

#create x where x column's values as float
x=df[df.columns].values.astype(float)
#create a minimum and maximum processor object
min_max_scaler=preprocessing.MinMaxScaler()
#create an object to transform the data to fit minimax processor
x_scaled=min_max_scaler.fit_transform(x)
#run the normalizer on the dataframe
df_normalized=pd.DataFrame(x_scaled,columns=df.columns)
#view the dataframe
df_normalized

data=['ad_id','xyzCampId','fbCampId','interest','Spent','ROAS','gender_F','costPerCon','gender_M' ,'age_30-34' ,'age_35-39' ,'age_40-44' ]
df_normalized1=df_normalized.loc[:,df.columns.isin(data)]
df_normalized1

X=df_normalized1.loc[:,df_normalized1.columns!='ROAS']
X.columns.values
print(X.shape)

Y=df_normalized1.loc[:,df_normalized1.columns=='ROAS']
Y.columns.values
print(Y.shape)


#******************************************************************************
#***séléction des features importants qui ont un impact sur le target**********


                         #****RFE*****
                         

from sklearn.feature_selection import RFE
from sklearn.linear_model import LinearRegression

logreg= LinearRegression()
rfe=RFE(logreg,7)
                                 
rfe=rfe.fit(X,Y.values.ravel())
print(rfe.support_)
print(rfe.ranking_)
print X.columns







            *****section exceptionnelle*****
list=rfe.ranking_
print list
for i in range(len(list)):
     if list[i]==1:

famille_panda_df = pd.DataFrame(rfe.ranking_,index = X.columns,columns=['rank'])
famille_panda_df


for  id_ligne,contenu_ligne in famille_panda_df.iterrows():
  if contenu_ligne['rank']==1: 
    print id_ligne

*******************************************************************************
 









     
cols1 = ['ad_id','fbCampId','interest','Spent','costPerCon','gender_M','age_40-44']
X1 = X [cols1] 
y1 = Y ['ROAS']

print X1
print y1
#****************implementing the model***********

import statsmodels.api as sm


logit_model1=sm.Logit(y1,X1)
result1=logit_model1.fit()
print(result1.summary())
dat=result1.summary()
famille_panda_df2 = pd.DataFrame(dat)
famille_panda_df


cols2= ['ad_id','fbCampId','Spent','costPerCon','gender_M'] 

X2 = X [cols2] 
y2 = Y ['ROAS']

 
#****************implementing the model***********

logit_model2 = sm.Logit (y2, X2) 
result2 = logit_model2.fit() 
print (result2.summary2 ())

#****************logistic regression model fitting***********************
                               
from sklearn.linear_model import LinearRegression
from sklearn import metrics
 X_train,X_test,Y_train,Y_test=train_test_split(X,Y,test_size=0.3,random_state=0)
logreg = LinearRegression () 
logreg.fit(X_train,Y_train)

#**************predicting the test set results and calculating the accuracy*****
Y_pred=logreg.predict(X_test)
print('accuracy of Logistic regression classifier on test set:{: .2f}'.format(logreg.score(X_test,Y_test)))



#***********confusion matrix*****************************

from sklearn.metrics import confusion_matrix

confusion_matrix=confusion_matrix(Y_test,Y_pred)
print(confusion_matrix)











data=df[['ad_id','fbCampId','Spent','costPerCon','gender_M']].drop_duplicates()
data.head()
data.shape
data.info()
data.shape




#**************************** Kmeans*****************************************

from sklearn.cluster import KMeans

#SEGMENTATION DES VARIABLES DE data en utilisant Kmean cluster

#identification du nombre de cluster important pour la segmentation


wcss = []
for i in range(1,11):
    km=KMeans(n_clusters=i,init='k-means++', max_iter=300, n_init=10, random_state=0)
    km.fit(data)
    wcss.append(km.inertia_)
plt.plot(range(1,11),wcss)
plt.title('Elbow Method')
plt.xlabel('Number of clusters')
plt.ylabel('wcss')
plt.show()


#la repartition des segments dans chaque cluster




cluster = KMeans(n_clusters=4)
data['cluster'] = cluster.fit_predict(data)
data['cluster'].value_counts()

#nous trouvons que la meilleure repartition ici est le 0 cluster cependant nous allons verifier 
#la clustering ou segmentation graphique par liner dimentionality reduction
#VERIFIONS


from sklearn.decomposition import PCA

pca = PCA(n_components=2)
data['data'] = pca.fit_transform(data)[:,0]
data['ROAS'] = pca.fit_transform(data)[:,1]
data = data.reset_index(drop=True)





sns.set_context("notebook", font_scale=1.1)
sns.set_style("ticks")
# Create scatterplot of dataframe
sns.lmplot('data', # Horizontal axis
           'ROAS', # Vertical axis
           data=data, # Data source
           fit_reg=False, # Don't fix a regression line
           hue="cluster", # Set color
           scatter_kws={"marker": "D", # Set marker style
                        "s": 100}) # S marker size
# Set title
plt.title('Histogram of data/ROAS')
# Set x-axis label
plt.xlabel('data')
# Set y-axis label
plt.ylabel('ROAS')


#nous constatons que le cluster3 est le meilleur segment d'identification de notre cible 
#POURTANT AVEC LA REPARTITION ON AVAIT 0 COMME MEILLEUR CLUSTER POUR LE CIBLAGE
#ENCORE VERIFIONS AVEC LES MOYENNES PAR GROUPBY



cluster_0 = data.loc[data['cluster']==0]
cluster_1 = data.loc[data['cluster']==1]
cluster_2 = data.loc[data['cluster']==2]
cluster_3 = data.loc[data['cluster']==3]


data = data.groupby('cluster').mean()
print data


max=np.max(data['ROAS'])
print max

for x in data.itertuples():
    if(x.ROAS==max):
      print (x)
     


