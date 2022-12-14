#%%
import preprocessing as prep
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, roc_auc_score, recall_score, precision_score
from sklearn.preprocessing import PolynomialFeatures
from sklearn.decomposition import PCA
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import GridSearchCV

X_train, X_test, y_train, y_test = prep.preprocess(seed=3)

#%%

kmax = 30
parameters = {'n_neighbors':np.arange(1,kmax)}

kNN = KNeighborsClassifier() # avec weights= 'distance' ca danne une bonne precision (93.2%) sans bc calculation avec 7 neighbours
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
print('precision:', precision_score(y_test, y_pred))
print(confusion_matrix(y_test, y_pred))

print("best parameter: ", model.best_params_)
print("best score: ", model.best_score_)  # but as far as I see, this is not a score related to the test-data !??
# %%

model = KNeighborsClassifier(n_neighbors=7)
model.fit(X_train, y_train)

#%%

y_pred = model.predict(X_test)

score_train = model.score(X_train, y_train)
score_test = model.score(X_test, y_test)
recall_test = recall_score(y_test, y_pred)

print('score_train:', score_train)
print('score_test:', score_test)
print('recall_test:', recall_test)
print('precision:', precision_score(y_test, y_pred))
print(confusion_matrix(y_test, y_pred))

# %%

accs = np.zeros(kmax)
recs = np.zeros(kmax)
precs = np.zeros(kmax)

for i in range(1,kmax):
    inn_model = KNeighborsClassifier(n_neighbors=i)
    inn_model.fit(X_train, y_train)

    inn_pred = inn_model.predict(X_test)

    score_train = inn_model.score(X_train, y_train)
    accs[i] = inn_model.score(X_test, y_test)
    recs[i] = recall_score(y_test, inn_pred)
    precs[i] = precision_score(y_test, inn_pred)

#%%

plt.plot(np.arange(2,kmax+1),accs[1:],label="accuracy",color = 'blue')
plt.plot(np.arange(2,kmax+1),recs[1:],label="recall",color='green')
plt.plot(np.arange(2,kmax+1),precs[1:],label="precision",color='purple')
plt.ylim(0.7,1.0)
plt.ylabel("scores")
plt.xlabel("number of neighbors")
plt.legend()
plt.savefig("Graphics/kNN.pdf")
plt.show()

# %%
