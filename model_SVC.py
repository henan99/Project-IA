import numpy as np
from sklearn.svm import SVC
import preprocessing as prep
from sklearn.model_selection import GridSearchCV

X_train, X_test, y_train, y_test = prep.X_train, prep.X_test, prep.y_train, prep.y_test

params = {'C': np.logspace(-1,2,20)}
svc = SVC()
model = GridSearchCV(svc, params)
model.fit(X_train, y_train)
score = model.score(X_test, y_test)

print(score)
print(model.best_estimator_, model.best_params_, model.best_score_)