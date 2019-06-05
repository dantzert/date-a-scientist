import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

#Create your df here:
df = pd.read_csv("profiles.csv")

#print(df.columns)
#print(df.body_type.value_counts())
#print(df.job.value_counts())
# predict body_type by job and alcohol consumption? or job by age and smokes/drinks/drugs?
percent_male = len(df[df.sex == 'm']) / len(df)

plt.pie([percent_male, 1-percent_male],labels = ['male','female'])


plt.show()

# Predict body_type by job and alcohol consumption

# format drinking as numbers
drink_mapping =  {"not at all": 0, "rarely": 1, "socially": 2, 
 "often": 3, "very often": 4, "desperately": 5}
df['drinks_code'] = df.drinks.map(drink_mapping)

# format jobs as numbers
job_mapping = {'other':0, 'student':1 ,'science / tech / engineering':2 ,
'computer / hardware / software' : 3, 
'artistic / musical / writer ' : 4  ,
'sales / marketing / biz dev' : 5  ,
'medicine / health ' : 6        ,   
'education / academia' : 7   ,
'executive / management' : 8    ,
'banking / financial / real estate':9   ,
'entertainment / media'    : 10    ,       
'law / legal services' : 11    ,
'hospitality / travel' : 12    ,            
'construction / craftsmanship' : 13    ,     
'clerical / administrative'   : 14     ,
'political / government' : 15         ,       
'rather not say' : 16              ,          
'transportation' : 17             ,           
'unemployed' : 18                    ,     
'retired'    : 19          ,                 
'military'  : 20     }
df['jobs_code'] = df.job.map(job_mapping)

#format body_type
body_mapping = { 'average' : 0 ,  
'fit' : 1               ,
'athletic' : 2        ,  
'thin' : 3             ,
'curvy'  : 4            , 
'a little extra' : 5   ,  
'skinny'  : 6           ,
'full figured' : 7       ,
'overweight' : 8        ,
'jacked' : 9           , 
'used up'  : 10      ,
'rather not say' : 11 }
df['body_code'] = df.body_type.map(body_mapping)

# normalize features

import sklearn.preprocessing as pp

feature_data = df[['jobs_code','drinks_code']]
x = feature_data.values
min_max_scaler = pp.MinMaxScaler()
x_scaled = min_max_scaler.fit_transform(x)
all_data = pd.DataFrame(x_scaled,columns = feature_data.columns)
all_data['body_code'] = df['body_code']
#print(all_data)
all_data.dropna(axis = 0,inplace = True)
#print(all_data)

# K nearest neighbors makes intuitive sense for this question
from sklearn.neighbors import KNeighborsClassifier
# later we'll loop through k values to find optimum accuracy
# classifier = KNeighborsClassifier(n_neighbors = 5)

# form X [features] and y [body_type] arrays
X = np.array(all_data[['jobs_code','drinks_code']])
#print(X)

y = np.array(all_data['body_code'])
#print(y)

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X,y,test_size = 0.2, random_state =11)

# the actual machine learning bit
#classifier.fit(X_train,y_train)

#print(classifier.score(X_test,y_test))

# 23.5 % accuracy with 5 neighbors. By chance would be 8.33% so there is definitely some predictive power

# Find the optimum value for k

# scores = []

#for k in range(100,200,1):
 #   classifier = KNeighborsClassifier(n_neighbors = k)
  #  classifier.fit(X_train,y_train)
   # scores.append(classifier.score(X_test,y_test))

#plt.plot(range(100,200,1),scores)
#plt.xlabel('n_neighbors')
#plt.ylabel('accuracy')
#plt.show()


#160 neighbors appears optimal
classifier = KNeighborsClassifier(n_neighbors = 160)
classifier.fit(X_train,y_train)
print(classifier.score(X_test,y_test)) # returns mean accuracy

# 29.457% accuracy with 160 neighbors. By chance would be 8.33%

# Score = 
# Precision = 
# Recall = 
