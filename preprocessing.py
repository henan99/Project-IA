import numpy as np
import pandas as pd
import keras
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

#imports data and returns it as two vectors for data and labels
def data_as_vector():
    pulsar_file = 'data/HTRU2/HTRU_2.csv' #filepath
    #import as dataframe
    pulsar_file_df = pd.DataFrame(pd.read_csv(pulsar_file, header=None))
    #add headers
    pulsar_file_df.columns = ['mean_ip', 'std_ip', 'kurtosis_ip', 'skewness_ip',
                            'mean_dm', 'std_dm', 'kurtosis_dm', 'skewness_dm',
                            'class']

    features = pulsar_file_df.columns

    featureVector = pulsar_file_df[features] #data vector
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

def scale_onehot_and_split(featureVector, targets):
    seed = 8+18+12 #fixed seed
    #fixed seed split 75-25
    X_train, X_test, y_train, y_test = train_test_split(featureVector, targets, test_size=0.25, random_state=seed)

    scale = StandardScaler()
    X_train = scale.fit_transform(X_train) # normalize data
    X_test = scale.transform(X_test)

    y_train_onehot = keras.utils.to_categorical(y_train)
    y_test_onehot = keras.utils.to_categorical(y_test)

    return X_train, X_test, y_train_onehot, y_test_onehot