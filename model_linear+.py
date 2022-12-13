#%%
import preprocessing as prep
from sklearn.linear_model import LogisticRegressionCV
import numpy as np
from sklearn.metrics import confusion_matrix, roc_auc_score, recall_score, precision_score
from sklearn.preprocessing import PolynomialFeatures
from sklearn.decomposition import PCA

X_train, X_test, y_train, y_test = prep.preprocess(seed=3)
poly = PolynomialFeatures(degree=2)
X_train = poly.fit_transform(X_train)
X_test = poly.transform(X_test)

""" pca = PCA(20)
X_train = pca.fit_transform(X_train)
X_test = pca.transform(X_test) """

#%%

C = np.logspace(-2, 2, 50)
model = LogisticRegressionCV(Cs = C, penalty='l2', max_iter=1000, class_weight='balanced')

model.fit(X_train, y_train)#, sample_weight = None)

#%%

y_pred = model.predict(X_test)

score_train = model.score(X_train, y_train, sample_weight = None)
score_test = model.score(X_test, y_test, sample_weight = None)
recall_test = recall_score(y_test, y_pred)

print('score_train:', score_train)
print('score_test:', score_test)
print('recall_test:', recall_test)
print('precision:', precision_score(y_test, y_pred))
print(confusion_matrix(y_test, y_pred))
print('best C:', model.C_)

# %%
