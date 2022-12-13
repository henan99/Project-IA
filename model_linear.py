
import preprocessing as prep
from sklearn.linear_model import LogisticRegressionCV
import numpy as np
from sklearn.metrics import confusion_matrix, roc_auc_score
from sklearn.preprocessing import PolynomialFeatures

X_train, X_test, y_train, y_test = prep.preprocess(seed=3)

C = np.logspace(-4, 4, 50)
model = LogisticRegressionCV(Cs = C, penalty='l2')

model.fit(X_train, y_train)#, sample_weight = None)
y_pred = model.predict(X_test)

score_train = model.score(X_train, y_train, sample_weight = None)
score_test = model.score(X_test, y_test, sample_weight = None)

print('score_train:', score_train)
print('score_test:', score_test)
print(confusion_matrix(y_test, y_pred))
print('best C:', model.C_)


