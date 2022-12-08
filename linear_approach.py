import preprocessing as prep
from sklearn.linear_model import LogisticRegressionCV
import numpy as np


C = np.logspace(-4, 4, 20)
model = LogisticRegressionCV(Cs = C, penalty='l2')
model.fit(prep.X_train, prep.y_train, sample_weight = None)
print(model.score(prep.X_train, prep.y_train, sample_weight = None))
score = model.score(prep.X_test, prep.y_test, sample_weight = None)
print('score:', score)
print('C:', model.C_)