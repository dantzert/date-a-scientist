import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

# bayes specific stuff
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB

#Create your df here:
df = pd.read_csv("profiles.csv")

#print(df.columns)
#print(df['religion'].value_counts())
#print(df.essay1[2])
#print(df.job.value_counts())

# Use Naive Bayes to predict occupation by essays

# so labels are job_codes
# just goes in as a list of codes, naive bayes will match by index
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
# jobs_code column is that list


# combine the text of all essays into one column
df['all_essays'] = df['essay0'] + df['essay1'] 
+ df['essay2'] + df['essay3'] + df['essay4'] + df['essay5'] 
+ df['essay6'] + df['essay7'] + df['essay8'] + df['essay9']

#print(df['all_essays'].head())

# do the actual training

counter = CountVectorizer()

#essays_list = list(df['all_essays'])

essay_counts = counter.fit_transform(df['all_essays'].values.astype('U'))

classifier = MultinomialNB()

df['jobs_code'].fillna(0,inplace = True)
labels = np.array(df['jobs_code'])
# record NA's as 0's (other and didn't fill out make sense to group together)

print(labels.shape)
print(essay_counts.shape)

#classifier.fit(essay_counts,training_labels)

# train_test split

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(essay_counts,labels,test_size = 0.2, random_state =11)

classifier.fit(X_train,y_train)

print(classifier.score(X_test,y_test)) # returns mean accuracy

# 40.5% accuracy. By chance would be 1/21 or 4.7% (about a 9x improvement over random)



