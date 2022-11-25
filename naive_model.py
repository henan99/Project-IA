import numpy as np

class NaiveModel:
    def __init__(self):
    #constructor (currently unused)
        print("constructed")

    def predict(self):
    #always predicts that it is not a pulsar
        return 0

    def score(self,X,y):
    #evaluates one-hot attributes on what proportion corresponds to naive assumtion
        #print(len(y)) #number of objects
        #print(np.sum(y[:,0])) #number of non-pulars (features passed as onehot)
        return np.sum(y[:,0])/len(y)