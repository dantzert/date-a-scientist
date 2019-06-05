import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

#Create your df here:
df = pd.read_csv("profiles.csv")

#print(df.columns)
# ------------------------------
# K Nearest Neighbors Regression
# ------------------------------
# predict income by age and height

# NORMALIZE FEATURES

import sklearn.preprocessing as pp

feature_data = df[['age','height']]
x = feature_data.values
min_max_scaler = pp.MinMaxScaler()
x_scaled = min_max_scaler.fit_transform(x)
all_data = pd.DataFrame(x_scaled,columns = feature_data.columns)
all_data['income'] = df['income']
print(len(all_data))
all_data = all_data[ all_data['income'] < 150000 ] 
# remove over $150k
all_data.dropna(axis = 0,inplace = True)
print(len(all_data))


# K nearest neighbors makes intuitive sense for this question
from sklearn.neighbors import KNeighborsRegressor
# later we'll loop through k values to find optimum accuracy
#regressor = KNeighborsRegressor(n_neighbors = 100)

# form X [features] and y [target] arrays
X = np.array(all_data[['age','height']])
#print(X)

y = np.array(all_data['income'])
#print(y)


from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X,y,test_size = 0.2, random_state = 292)


#regressor.fit(X_train,y_train)

#print(regressor.score(X_test,y_test))

# Find the optimum value for k

scores = []

#for k in range(250,1750,100):
 #   regressor = KNeighborsRegressor(n_neighbors = k)
  #  regressor.fit(X_train,y_train)
   # scores.append(regressor.score(X_test,y_test))

#plt.figure()
#plt.plot(range(250,1750,100),scores)
#plt.xlabel('n_neighbors')
#plt.ylabel('score')
#plt.show()

# it's  1000
k_optimum = 1000
regressor = KNeighborsRegressor(n_neighbors = k_optimum)
regressor.fit(X_train,y_train)
y_predict = regressor.predict(X_test)


print(regressor.score(X_test,y_test))

# Score (R^2 coefficient of determination) = 0.016940736811577706


plt.figure()
plt.plot(range(150000),range(150000))
plt.scatter(y_test,y_predict,alpha = 0.1)
plt.xlabel('actual income')
plt.ylabel('predicted income')
plt.show()
