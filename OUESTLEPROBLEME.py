# Ou est le problème ??? C'est pas possible que le modele est deja parfait, non ??? ://

import preprocessing as prep
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegressionCV
import numpy as np

X, y = prep.data_as_vector()
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25)
skalar=StandardScaler()
X_train = skalar.fit_transform(X_train)
X_test = skalar.transform(X_test)

C = np.logspace(-4, 4, 20)
model = LogisticRegressionCV(Cs = C, penalty='l2')
model.fit(X_train, y_train, sample_weight = None)
score = model.score(X_test, y_test, sample_weight = None)

print('score:', score)


