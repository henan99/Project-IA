import preprocessing as prep
from sklearn.linear_model import LogisticRegressionCV

X_train, X_test, y_train, y_test = prep.preprocess()

model = LogisticRegressionCV()