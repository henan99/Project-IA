import numpy as np
import pandas as pd
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

def scale_and_split(featureVector, targets):
    seed = 8+18+12 #fixed seed

    scale = StandardScaler()
    featureVector = scale.fit_transform(featureVector) # normalize data

    #fixed seed split 75-25
    X_train, X_test, y_train, y_test = train_test_split(featureVector, targets, test_size=0.25, random_state=seed)

    return X_train, X_test, y_train, y_test