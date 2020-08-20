import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt 

#ignore double quotes - 3
dataset = pd.read_csv('Restaurant_Reviews.tsv', delimiter = '\t', quoting = 3) 

import re
import nltk 
nltk.download('stopwords')
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer

corpus = []
for i in range(0, 1000):
    review = re.sub('[^a-zA-Z]', ' ', dataset['Review'][i])
    review = review.lower()
    review = review.split()
    
    all_stopwords = stopwords.words('english')
    all_stopwords.remove('not')

    #applying spemming 
    ps = PorterStemmer()
    review = [ps.stem(word) for word in review if not word in set(all_stopwords)]

    review = ' '.join(review)
    corpus.append(review)

from sklearn.feature_extraction.text import CountVectorizer
cv = CountVectorizer(max_features = 1300)
X = cv.fit_transform(corpus).toarray()
y = dataset.iloc[:, -1].values

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.20, random_state = 0)

'''
from sklearn.naive_bayes import GaussianNB
classifier = GaussianNB()
classifier.fit(X_train, y_train)
0.73 - 1300, 1500 max_features
'''

''' 
from sklearn.tree import DecisionTreeClassifier
classifier = DecisionTreeClassifier(criterion = 'entropy', random_state = 0)
classifier.fit(X_train, y_train)
0.72 - 1300 max_features
0.75 - 1500 max_features
'''

from sklearn.ensemble import RandomForestClassifier
classifier = RandomForestClassifier(n_estimators = 10, criterion = 'entropy', random_state = 0)
classifier.fit(X_train, y_train)
'''
0.725 - 10 n_estimators, 1500 max_features | 10 n_estimators, 1200 max_features 
0.755 - 10 n_estimators, 1300 max_features, criterion = entropy 
0.765 - 10 n_estimators, 1300 max_features, criterion = gini 
'''

y_pred = classifier.predict(X_test)
    
from sklearn.metrics import confusion_matrix, accuracy_score
cm = confusion_matrix(y_test, y_pred)
print(accuracy_score(y_test, y_pred))