#%%
import preprocessing as prep
import numpy as np
from sklearn.metrics import confusion_matrix, roc_auc_score, recall_score
from sklearn.preprocessing import PolynomialFeatures
from sklearn.decomposition import PCA
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import GridSearchCV

#%%
X_train, X_test, y_train, y_test = prep.preprocess(seed=3)

kmax = 30
parameters = {'n_neighbors':np.arange(1,kmax)}

kNN = KNeighborsClassifier()
model = GridSearchCV(kNN, parameters)
model.fit(X_train, y_train)
#%%

y_pred = model.predict(X_test)

score_train = model.score(X_train, y_train)
score_test = model.score(X_test, y_test)
recall_test = recall_score(y_test, y_pred)

print('score_train:', score_train)
print('score_test:', score_test)
print('recall_test:', recall_test)
print(confusion_matrix(y_test, y_pred))

print("best parapeter: ", model.best_params_)
print("best score: ", model.best_score_)
# %%
