import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

def scale_and_split(featureVector, targets):
    seed = 8+18+12

    scale = StandardScaler()
    featureVector = scale.fit_transform(featureVector) # normalize data

    X_train, X_test, y_train, y_test = train_test_split(featureVector, targets, test_size=0.25, random_state=seed)

    return X_train, X_test, y_train, y_test