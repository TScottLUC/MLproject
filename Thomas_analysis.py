from preprocessing import preprocess_data
import pandas as pd
from sklearn.dummy import DummyClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import GridSearchCV # Implements Grid Search and Cross-validation
from sklearn.metrics import f1_score

# Read in the preprocessed data
dataList = preprocess_data()

x_train = dataList[0]
x_test = dataList[1]
y_train = dataList[2]
y_test = dataList[3]

# Drop date - datetime does not work with some classifiers and splitting to month/day/year did not improve model performance
x_train = x_train.drop(['Date'], axis=1)
x_test = x_test.drop(['Date'], axis=1)

# Baseline

print("Baseline:")

# Define strategies to try
parameters = {'strategy': ('most_frequent', 'stratified', 'prior', 'uniform')}

# Grid search, fit, then calculate accuracy and F1 score on test set
dummy = DummyClassifier()
dummy = GridSearchCV(dummy, parameters, cv=10, scoring=['accuracy', 'f1'], refit='f1')
dummy.fit(x_train, y_train)
dummyData = pd.DataFrame(dummy.cv_results_)

bestDummy = dummy.best_estimator_

print("Best dummy classifier: " + str(bestDummy))
print("Accuracy on test set: " + str(bestDummy.score(x_test, y_test)))
print("F1 score on test set: " + str(f1_score(y_test, bestDummy.predict(x_test))))

print('--------')

# Logistic Regression

print("Logistic Regression:")

# Define parameters to test for penalty, C, fit_intercept, and class_weight
parameters = {'penalty': ('l1', 'l2'), 'C': [0.00001, 0.0001, 0.001, 0.01, 0.1, 1, 10, 100, 1000, 10000],
              'fit_intercept': (False, True), 'class_weight': ('balanced', None), 'solver': ['liblinear']}

# Grid search, fit, then calculate accuracy and F1 score on test set
logistic = LogisticRegression()
logistic = GridSearchCV(logistic, parameters, cv=10, scoring=['accuracy', 'f1'], refit='f1')
logistic.fit(x_train, y_train)
lData = pd.DataFrame(logistic.cv_results_)

print("Best logistic regression: " + str(logistic.best_estimator_))
print("Accuracy on test set: " + str(logistic.best_estimator_.score(x_test, y_test)))
print("F1 score on test set: " + str(f1_score(y_test, logistic.best_estimator_.predict(x_test))))

print('--------')

# Naive bayes

print("Naive bayes:")

# No parameters, no need to use grid search

# Only need to fit and calculate accuracy and F1 score
gaussian = GaussianNB()
gaussian.fit(x_train, y_train)

print("Accuracy on test set: " + str(gaussian.score(x_test, y_test)))
print("F1 score on test set: " + str(f1_score(y_test, gaussian.predict(x_test))))
