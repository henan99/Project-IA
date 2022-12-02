# Ou est le problème ??? C'est pas possible que le modele est deja parfait, non ??? ://

import preprocessing as prep
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegressionCV
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_curve, roc_auc_score
from sklearn.model_selection import cross_val_score, cross_val_predict
import numpy as np

X, y = prep.data_as_vector()
skalar=StandardScaler()
X = skalar.fit_transform(X)


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.5)

RFC = RandomForestClassifier(n_jobs=2,n_estimators=10)
RFC.fit(X_train,y_train)
rfc_predict = RFC.predict(X_test)
print(confusion_matrix(y_test, rfc_predict))
score = RFC.score(X_test, y_test)
print('scoreRFC:', score)

C = np.logspace(-1, 1, 20)
model = LogisticRegressionCV( Cs=C, penalty='l2')
model.fit(X_train, y_train)
coefs = model.coef_

y_pred = model.predict(X_test)
score = model.score(X_test, y_test)
print(confusion_matrix(y_test, y_pred))

print('scoreLR:', score)


