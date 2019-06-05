import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

#Create your df here:
df = pd.read_csv("profiles.csv")

#print(df.columns)

#print(df.body_type.value_counts())
#print(df.job.value_counts())

# regression
# predict income from height, age, and sex

# MLR, height and age are naturally continuous
# sex is  binary

#print(df.income.value_counts())
# income has a ton of "-1"s I'll want to remove

# print(df.height)
# in inches, seems to be well filled

#print(df.age)
# in years, seems well filled

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

# format sex as numbers
sex_mapping =  {"f": 0, "m": 1}
df['sex_code'] = df.sex.map(sex_mapping)


# remove -1's
df = df[df.income != -1]
# remove over $150k
df = df[df.income < 150000]
print(df.income.value_counts())

x = df[['age','height','sex_code']]
y = df[['income']]

x_train,x_test, y_train, y_test = train_test_split(x,y,test_size = 0.2, random_state =11)

lm = LinearRegression()


model = lm.fit(x_train,y_train)

y_predict = lm.predict(x_test)



#print("Train score:")
#print(lm.score(x_train, y_train))

print("Test score:")
print(lm.score(x_test, y_test))
print(lm.coef_)
# Score (R^2 coefficient of determination) = 0.16688290203737888

plt.scatter(y_test,y_predict,alpha = 0.1)
plt.plot(range(100000),range(100000))
plt.xlabel('Actual Income')
plt.ylabel('Predicted Income')

plt.show()


