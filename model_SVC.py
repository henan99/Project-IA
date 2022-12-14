
# needs about a minute to compile, score > 0.980
#%%
import numpy as np
from sklearn.svm import SVC
import preprocessing as prep
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import confusion_matrix, roc_auc_score, recall_score, precision_score

X_train, X_test, y_train, y_test = prep.preprocess(seed=3)
#%%

params1 = {'C': np.logspace(-1,2,0)}
params2 = {'C': np.logspace(-1,2,20), 'kernel':['linear', 'poly', 'rbf', 'sigmoid']}
#   Grid search with parmas2 as parameters returns as best parameters: {'C': 16.23776739188721, 'kernel': 'rbf'}
params3 = {'C': [0.5, 16, 50], 'kernel':['linear', 'poly', 'rbf', 'sigmoid'], 'gamma': ['scale', 'auto']}
params4 = {'C': np.logspace(-1,2,20), 'degree': [2, 3, 4, 5, 6]}
#   running with params4 will need more than 1 hour to compile and resolt in parmas {'C': 1.2742749857031335, 'degree': 3} like the standart before.
svc = SVC(class_weight='balanced', kernel= 'rbf') # higher score but significantly lower recall with class_weight = NONE.
model = GridSearchCV(svc, params1)
model.fit(X_train, y_train)

#%%
# this model gives a recall of 89.9 % !!
model = SVC(C=1.27, kernel='rbf', class_weight='balanced') # balance makes the smaller class more weighted
model.fit(X_train, y_train)

#%%
score = model.score(X_test, y_test)
y_pred = model.predict(X_test)
recall_test = recall_score(y_test, y_pred)

print('train_score', model.score(X_train, y_train))
print('testscore', score)
print('recall:', recall_test)
print('precision:', precision_score(y_test, y_pred))
print(model.best_estimator_, model.best_params_)#,  model.best_score_)
print(confusion_matrix(y_test, y_pred))
svc_best = model.best_estimator_
print('number of support V:', svc_best.n_support_)

# %%
# Evaluation for a singel model

score = model.score(X_test, y_test)
y_pred = model.predict(X_test)
recall_test = recall_score(y_test, y_pred)

print('train_score', model.score(X_train, y_train))
print('testscore', score)
print('recall:', recall_test)
print('precision:', precision_score(y_test, y_pred))
print(confusion_matrix(y_test, y_pred))
print('number of support V:', model.n_support_)
# %%
