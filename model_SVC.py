
# needs about a minute to compile, score > 0.980
#%%
import numpy as np
from sklearn.svm import SVC
import preprocessing as prep
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import confusion_matrix, roc_auc_score, recall_score, precision_score

X_train, X_test, y_train, y_test = prep.preprocess(seed=3)
#%%

params1 = {'C': np.logspace(-1,2,20)}
params2 = {'C': np.logspace(-1,2,20), 'kernel':['linear', 'poly', 'rbf', 'sigmoid']}
#   Grid search with parmas2 as parameters returns as best parameters: {'C': 16.23776739188721, 'kernel': 'rbf'}

svc = SVC()
model = GridSearchCV(svc, params1)
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
# %%
