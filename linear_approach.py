import preprocessing as prep
from sklearn.linear_model import LogisticRegressionCV
import numpy as np

X_train, X_test, y_train, y_test = prep.preprocess_without_onehot(seed=1)

C = np.logspace(-4, 4, 20)
model = LogisticRegressionCV(Cs = C, penalty='l2')
model.fit(X_train, y_train, sample_weight = None)
print(model.score(X_train, y_train, sample_weight = None))
score = model.score(X_test, y_test, sample_weight = None)
print('score:', score)