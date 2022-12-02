import numpy as np
import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt
import numpy.polynomial.polynomial as poly
from sklearn.preprocessing import StandardScaler

from sklearn.model_selection import train_test_split


pulsar_file = 'data/HTRU2/HTRU_2.csv'


pulsar_file_df = pd.DataFrame(pd.read_csv(pulsar_file, header=None))
pulsar_file_df.columns = ['mean_ip', 'std_ip', 'kurtosis_ip', 'skewness_ip',
                          'mean_dm', 'std_dm', 'kurtosis_dm', 'skewness_dm',
                          'class']







features = pulsar_file_df.columns

featureVector = pulsar_file_df[features]
targets = pulsar_file_df['class']

scale = StandardScaler()
featureVector = scale.fit_transform(featureVector)


xTrain, xTest, yTrain, yTest = train_test_split(featureVector, targets, test_size = 0.15, random_state = 42)
m,n = xTrain.shape

# defining model
simpleAnn = tf.keras.models.Sequential([
    tf.keras.layers.Input(shape = (n, )),
    tf.keras.layers.Dense(32, activation = 'relu'),
    tf.keras.layers.Dense(64, activation = 'relu'),
    tf.keras.layers.Dropout(0.07),
    tf.keras.layers.Dense(128, activation = 'relu'),
    tf.keras.layers.Dense(1, activation = 'sigmoid')
])