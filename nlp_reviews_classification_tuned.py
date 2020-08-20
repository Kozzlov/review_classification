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

0.73
accuracy: 67.25
standard diviation: 5.30
'''

'''
from sklearn.tree import DecisionTreeClassifier
classifier = DecisionTreeClassifier(criterion = 'entropy', random_state = 0)
classifier.fit(X_train, y_train)

0.72
accuracy: 75.87
standard diviation: 4.47
'''


from sklearn.ensemble import RandomForestClassifier
classifier = RandomForestClassifier(n_estimators = 10, 
                                    criterion = 'entropy',
                                    random_state = 0)
classifier.fit(X_train, y_train)

'''
0.755
accuracy: 77.62
standard diviation: 2.53
'''

y_pred = classifier.predict(X_test)
    
from sklearn.metrics import confusion_matrix, accuracy_score
from sklearn.model_selection import cross_val_score, GridSearchCV

cm = confusion_matrix(y_test, y_pred)
print(accuracy_score(y_test, y_pred))

accuracies = cross_val_score(estimator = classifier, X = X_train, y = y_train, cv = 10)
print('accuracy: {:.2f}'.format(accuracies.mean()*100))
print('standard diviation: {:.2f}'.format(accuracies.std()*100))

parameters = {'max_depth': [10, 20, 30, None],
              'min_samples_leaf': [1, 2, 4],
              'min_samples_split': [2, 4],
              'n_estimators': [20, 40, 60]}

grid_search = GridSearchCV(estimator = classifier,
                           param_grid = parameters,
                           scoring = 'accuracy',
                           cv = 10, 
                           n_jobs = -1) #use all processors

grid_search.fit(X_train, y_train)
best_accuracy = grid_search.best_score_
best_parameters = grid_search.best_params_
print('best accuracy: {:.2f}'.format(best_accuracy.mean()*100))
print('best parameters: ', best_parameters)
