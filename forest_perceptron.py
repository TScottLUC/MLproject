import pandas as pd
import numpy as np
from sklearn import linear_model
from sklearn.ensemble import ExtraTreesClassifier, RandomForestClassifier
from sklearn.decomposition import PCA
from sklearn import metrics
from catboost import CatBoostClassifier, Pool, cv
import matplotlib.pyplot as plt
from preprocessing import preprocess_data
import math
x_train, x_test, y_train, y_test = preprocess_data()

x_train = x_train.drop('Date', axis=1)
x_test = x_test.drop('Date', axis=1)

print(x_train.iloc[0, :])
print()
print(x_train.iloc[0, :].apply(lambda a : type(a)))

# EDA plots
categoricals = x_train.loc[:,]
plt.violinplot(x_train.loc[:,['RainToday', 'deltaWindSpeed', 'WindDirDelta']])


# LASSO for variable selection
lasso_model = linear_model.LassoCV(tol=100).fit(x_train, y_train)
lasso_model.coef_
non_zero_x = x_train.loc[:,lasso_model.coef_ != 0]
lasso_model_preds = 1*(lasso_model.predict(x_test) > 0.5)
print("LASSO F1:", metrics.f1_score(y_test, lasso_model_preds))
print("LASSO Accuracy:", metrics.accuracy_score(y_test, lasso_model_preds))

# Random forest model (note: try binning for continuous valued features)
model = ExtraTreesClassifier(n_estimators=2000, max_depth=10,
                                         criterion='entropy',
                                         max_features='sqrt',
                                         bootstrap=True)
model.fit(non_zero_x.values, y_train.values.reshape([len(y_train),]))
preds = model.predict(x_test.loc[:,lasso_model.coef_ != 0])
print("Random Forest F1:", metrics.f1_score(y_test, preds))
print("Random Forest Accuracy:", metrics.accuracy_score(y_test, preds))

# Variable importance plot
importances = model.feature_importances_
std = np.std([tree.feature_importances_ for tree in model.estimators_], axis=0)
forest_importances = pd.Series(importances, index=non_zero_x.columns)
fig, ax = plt.subplots()
forest_importances.plot.barh(yerr=std, ax=ax)
ax.set_title("Variable importance plot")
ax.set_ylabel("Mean decrease in impurity")
fig.tight_layout()

# Random forest grid search
depths = range(5, 11)
criterions = ['gini', 'entropy']
features = ['sqrt', 'log2', 10, 15, 20]
rf_grid = pd.DataFrame(columns = ['Depth', 'Criterion', 'Feature', 'F1', 'Accuracy'])
for depth in depths:
    for criterion in criterions:
        for feature in features:
            rf_model = RandomForestClassifier(n_estimators = 1000,
                                              criterion = criterion,
                                              max_depth = depth,
                                              max_features = feature)
            rf_model.fit(x_train, y_train)
            rf_preds = rf_model.predict(x_test)
            f1 = metrics.f1_score(y_test, rf_preds)
            acc = metrics.accuracy_score(y_test, rf_preds)
            rf_grid.append({'Depth' : depth, 'Criterion': criterion, 'Feature': feature, 
                            'F1': f1, 'Accuracy': acc}, ignore_index=True)
            print(rf_grid)
rf_grid

# SVM (0.522)
# svm_model = svm.SVC().fit(x_train, y_train)
# svm_preds = svm_model.predict(x_test)
# print("SVM F1:", metrics.f1_score(y_test, svm_preds))

# PCA
pca = PCA(n_components=20)
x_train_pc = pca.fit_transform(x_train)
x_test_pc = pca.fit_transform(x_test)

# Catboost
train_pool = catboost_pool = Pool(x_train, y_train)
test_pool = catboost_pool = Pool(x_test, y_test)
cat_model = CatBoostClassifier(iterations=2500, depth=8, learning_rate=0.03,
                               loss_function='CrossEntropy',
                               verbose=True)
cat_model.fit(train_pool)
cat_preds = cat_model.predict(test_pool)
print("Catboost F1:", metrics.f1_score(y_test, cat_preds))
print("Catboost Accuracy:", metrics.accuracy_score(y_test, cat_preds))

# Feature importance
plt.figure(figsize=(9, 15))
cat_results = pd.DataFrame({'Feature': cat_model.feature_names_, 'Importance':cat_model.feature_importances_})
cat_results.sort_values(by='Importance', inplace=True)
plt.barh(cat_results['Feature'], cat_results['Importance'])

# Confusion matrix
cm = metrics.confusion_matrix(y_test, cat_preds)
disp = metrics.ConfusionMatrixDisplay(confusion_matrix=cm)
disp.plot()
plt.show()

# Catboost grid search
cat_model_grid = CatBoostClassifier(iterations=2500, loss_function='CrossEntropy')
grid = {'learning_rate': np.arange(0.01, 0.11, 0.01),
        'depth': np.arange(1, 11, 1)}
grid_results = cat_model_grid.grid_search(grid, x_train, y_train, cv=5, stratified=True)
grid_results['params'] # Depth = 8, learning_rate = 0.03

# Catboost cross validation
params = {'iterations': 2000,
          'depth': 8,
          'loss_function': 'CrossEntropy',
          'verbose': False,
          'prediction_type': 'Class'
          }
cat_cv = cv(train_pool, params, nfold=5, stratified=True,
            as_pandas=True, type='TimeSeries', return_models=True)
cat_cv_model = cat_cv[1][3]
cat_cv_preds = cat_cv_model.predict(test_pool) # fix this later
print("Catboost F1:", metrics.f1_score(y_test, cat_cv_preds))
print("Catboost Accuracy:", metrics.accuracy_score(y_test, cat_cv_preds))

# Perceptron
perceptron = linear_model.Perceptron(penalty='l1').fit(x_train, y_train)
perceptron_preds = perceptron.predict(x_test)
print("Perceptron F1:", metrics.f1_score(y_test, perceptron_preds))
print("Perceptron Accuracy:", metrics.accuracy_score(y_test, perceptron_preds))

# Logistic regression
logit = linear_model.LogisticRegression(solver='saga', penalty='elasticnet', 
                                        l1_ratio=.25, max_iter=2500, tol=0.001).fit(x_train, y_train)
logit_preds = logit.predict(x_test)
print("Logistic Regression F1:", metrics.f1_score(y_test, logit_preds))
print("Logistic Regression Accuracy:", metrics.accuracy_score(y_test, logit_preds))

