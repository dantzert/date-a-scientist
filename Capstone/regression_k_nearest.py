import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

#Create your df here:
df = pd.read_csv("profiles.csv")

print(df.columns)
#print(df.body_type.value_counts())
#print(df.job.value_counts())


# K Nearest Neighbors Regression


# predict height by ethnicity, sex, body_type

#print(df.ethnicity.value_counts())
# i'll ignore ethnicities with frequencies of less than 1000
#print(df.sex.value_counts())
#print(df.body_type.value_counts())


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


# format sex as numbers
sex_mapping =  {"f": 0, "m": 1}
df['sex_code'] = df.sex.map(sex_mapping)

# format ethnicity
ethnic_mapping = { 'white':0,                                                              
'asian' : 1 ,     
'hispanic / latin' : 2 , 
'black' : 3           ,
'other' : 4 ,    
'hispanic / latin, white' : 5 ,     
'indian' : 6 }
df['ethnicity_code'] = df.ethnicity.map(ethnic_mapping)

# ignore rare ethnicity combinations
df = df[df.ethnicity_code < 7] # a bit messy, but gets rid of the non-ints

# normalize features

import sklearn.preprocessing as pp

feature_data = df[['body_code','sex_code','ethnicity_code']]
x = feature_data.values
min_max_scaler = pp.MinMaxScaler()
x_scaled = min_max_scaler.fit_transform(x)
all_data = pd.DataFrame(x_scaled,columns = feature_data.columns)
all_data['height'] = df['height']
#print(all_data)
all_data.dropna(axis = 0,inplace = True)
#print(all_data)


# K nearest neighbors makes intuitive sense for this question
from sklearn.neighbors import KNeighborsRegressor
# later we'll loop through k values to find optimum accuracy
# regressor = KNeighborsRegressor(n_neighbors = 1000)

# form X [features] and y [target] arrays
X = np.array(all_data[['sex_code','body_code','ethnicity_code']])
#print(X)

y = np.array(all_data['height'])
#y = y.reshape(-1,1)
#y = min_max_scaler.fit_transform(y) # normalize height as well
#y = y.reshape(-1,1)
#print(y)


from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X,y,test_size = 0.1, random_state =19)


# the actual machine learning bit
#regressor.fit(X_train,y_train)

#print(regressor.score(X_test,y_test))


# Find the optimum value for k

scores = []

for k in range(100,10000,500):
    regressor = KNeighborsRegressor(n_neighbors = k)
    regressor.fit(X_train,y_train)
    scores.append(regressor.score(X_test,y_test))
    print(k/10000)

plt.plot(range(100,10000,500),scores)
plt.xlabel('n_neighbors')
plt.ylabel('score')
plt.show()

# this isn't working at all
# hmmmm.....