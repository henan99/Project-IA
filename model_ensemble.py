
# the idea of this file ist to create a model in combining several models 
# of two classes. One class with a very high percision and one with a very high recall.
# the propositon for the first class was the SVC(c = 16, kernel='rbf') 
# and for the second class the SVC(c = 1.27, kernel='rbf', class_weight='balanced').

#%%
import numpy as np
from sklearn.svm import SVC
import preprocessing as prep
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, roc_auc_score, recall_score, precision_score

X_train, X_test, y_train, y_test = prep.preprocess(seed=3)
#%%

seeds = np.arange(0,11)
models = np.zeros(len(seeds))
for i in range(len(seeds)):
    X_train, X_test, y_train, y_test = train_test_split(X_train, y_train, random_state=seeds[i])
    model = SVC(C=1.27, kernel='rbf', class_weight='balanced') # balance makes the smaller class more weighted
    model.fit(X_train, y_train)
    models[i] = model

#%%
class ensemble:
    def __init__(self):
        print("constructed")

    def predict(self):
        return 0

    def score(self,X,y):
        return np.sum(y[:,0])/len(y)





