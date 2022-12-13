import numpy as np
import pandas as pd
import keras
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline

#imports data and returns it as two vectors for data and labels
def data_as_vector():
    pulsar_file = 'data/HTRU2/HTRU_2.csv' #filepath
    #import as dataframe
    pulsar_file_df = pd.DataFrame(pd.read_csv(pulsar_file, header=None))
    #add headers
    pulsar_file_df.columns = ['mean_ip', 'std_ip', 'kurtosis_ip', 'skewness_ip',
                            'mean_dm', 'std_dm', 'kurtosis_dm', 'skewness_dm',"class"]

    features = pulsar_file_df.columns
    
    featureVector = pulsar_file_df[features[:-1]] #data vector
    targets = pulsar_file_df['class'] #labels
    
    return featureVector, targets

def check_missing():
    #chercher des données manquantes
    X,y = data_as_vector()

    if(np.sum(~np.isfinite(X)).aggregate(np.sum) == 0): # aggregate sums over different pandas categories
        print("no missing data in attributes")
    else:
        print("there are attributes missing")
    if(np.sum(~np.isfinite(y)) == 0): # True wherever pos-inf or neg-inf or nan
        print("no lables missing")
    else:
        print("missing labels")

def tt_split(featureVector, targets, seed = 8+18+12):
    X_train, X_test, y_train, y_test = train_test_split(featureVector, targets, test_size=0.25, random_state=seed)
    return X_train, X_test, y_train, y_test


def pipeline(X_train, X_test, y_train, y_test):
    # pipeline for X: scale
    pipeX = Pipeline([('scaler', StandardScaler())])
    X_train = pipeX.fit_transform(X_train) # normalize data
    X_test = pipeX.transform(X_test)

    return X_train, X_test, y_train, y_test


# returns X_train, X_test, y_train, y_test after the preprocessing process of 'pipline'

def preprocess(seed = 8+18+12):
    data1, data2 = data_as_vector()
    X_train, X_test, y_train, y_test = tt_split(data1, data2, seed = seed)
    return pipeline(X_train, X_test, y_train, y_test)

X_train, X_test, y_train, y_test = preprocess(seed=3)
