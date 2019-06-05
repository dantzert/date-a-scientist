import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

#Create your df here:
df = pd.read_csv("profiles.csv")

#print(df.columns)
# ------------------------------
# K Nearest Neighbors Regression
# ------------------------------
# predict drinking by sex, smokes, drugs, diet

# FEAUTRES formatting

# format sex as numbers
sex_mapping =  {"f": 0, "m": 1}
df['sex_code'] = df.sex.map(sex_mapping)

# format smokes as numbers

smoke_map = { 'no':0 , 
'sometimes':1 ,   
'when drinking':2 ,
'yes' :3,
'trying to quit' : 4 }
df['smoke_code'] = df.smokes.map(smoke_map)

# format drugs as numbers

drugs_map = { 'never':0, 
'sometimes' : 1 ,
'often' : 2 }
df['drugs_code'] = df.drugs.map(drugs_map)

# format diet

diet_map = { 'mostly anything':0, 
'anything' : 1,     
'strictly anything':2, 
'mostly vegetarian':3 ,
'mostly other' : 4 ,  
'strictly vegetarian' : 5 ,
'vegetarian' : 6  ,  
'strictly other' : 7, 
'mostly vegan' : 8, 
'other' : 9 ,   
'strictly vegan' : 10,  
'vegan' : 11 ,   
'mostly kosher' : 12 , 
'mostly halal' : 13 , 'strictly halal' : 14 ,   
'strictly kosher' : 15,
'halal' : 16,  
'kosher' : 17}
df['diet_code'] = df.diet.map(diet_map)

# TARGET FORMAT

# format drinking as numbers
drink_mapping =  {"not at all": 0, "rarely": 1, "socially": 2, "often": 3, "very often": 4, "desperately": 5}
df['drinks_code'] = df.drinks.map(drink_mapping)


# NORMALIZE FEATURES

import sklearn.preprocessing as pp

feature_data = df[['sex_code','smoke_code','drugs_code','diet_code']]
x = feature_data.values
min_max_scaler = pp.MinMaxScaler()
x_scaled = min_max_scaler.fit_transform(x)
all_data = pd.DataFrame(x_scaled,columns = feature_data.columns)
all_data['drinks_code'] = df['drinks_code']
#print(all_data)
all_data.dropna(axis = 0,inplace = True)
#print(all_data)


# K nearest neighbors makes intuitive sense for this question
from sklearn.neighbors import KNeighborsRegressor
# later we'll loop through k values to find optimum accuracy
#regressor = KNeighborsRegressor(n_neighbors = 100)

# form X [features] and y [target] arrays
X = np.array(all_data[['sex_code','smoke_code','drugs_code','diet_code']])
#print(X)

y = np.array(all_data['drinks_code'])
#print(y)


from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X,y,test_size = 0.2, random_state = 292)


#regressor.fit(X_train,y_train)

#print(regressor.score(X_test,y_test))

# Find the optimum value for k

scores = []

#for k in range(250,350,10):
#    regressor = KNeighborsRegressor(n_neighbors = k)
#    regressor.fit(X_train,y_train)
#    scores.append(regressor.score(X_test,y_test))

#plt.figure()
#plt.plot(range(250,350,10),scores)
#plt.xlabel('n_neighbors')
#plt.ylabel('score')
#plt.show()

# it's somewhere around 280
k_optimum = 280
regressor = KNeighborsRegressor(n_neighbors = k_optimum)
regressor.fit(X_train,y_train)
y_predict = regressor.predict(X_test)

plt.figure()
plt.plot(range(5),range(5))
plt.scatter(y_test,y_predict,alpha = 0.1)
plt.xlabel('actual drinking')
plt.ylabel('predicted drinking')
plt.show()

print(regressor.score(X_test,y_test))