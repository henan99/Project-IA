import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy.polynomial.polynomial as poly
from sklearn.preprocessing import StandardScaler
from  sklearn.decomposition import PCA 
import tensorflow as tf 
from sklearn.model_selection import train_test_split


pulsar_file = 'data/HTRU2/HTRU_2.csv'


pulsar_file_df = pd.DataFrame(pd.read_csv(pulsar_file, header=None))
pulsar_file_df.columns = ['mean_ip', 'std_ip', 'kurtosis_ip', 'skewness_ip',
                          'mean_dm', 'std_dm', 'kurtosis_dm', 'skewness_dm',
                          'class']

#print(pulsar_file_df)


#print(pulsar_file_df.head(10)[pulsar_file_df["class"]==0])

#print(test[test["class"]==1])

#print(pulsar_file_df.drop(['class'], axis=1).corr())

#plt.figure(figsize=(8,6))
#_= sns.heatmap(pulsar_file_df.drop(['class'], axis=1).corr(), annot=True)
#plt.show()
fig = plt.figure()
_ = sns.pairplot(pulsar_file_df, kind='scatter', hue='class', plot_kws=dict(ec='w'))
plt.show()#
fig.savefig("pairplot_data.png")

# doing PCA


features = pulsar_file_df.columns

featureVector = pulsar_file_df[features]
targets = pulsar_file_df['class']

scale = StandardScaler()
featureVector = scale.fit_transform(featureVector)


pca = PCA()
pca.fit(featureVector)
featureVectorT = pca.transform(featureVector)

pcavecs = pca.components_
pcavals = pca.explained_variance_ 
pcavalsratio = pca.explained_variance_ratio_ 


plt.figure(1,(12,7))
plt.plot(featureVectorT[targets ==0,0],featureVectorT[targets ==0,1],"x",label ="0")
plt.plot(featureVectorT[targets ==1,0],featureVectorT[targets ==1,1],"x",label = "1")
plt.quiver(pcavecs[0,0],pcavecs[0,1],scale=2)
plt.quiver(pcavecs[1,0],pcavecs[1,1],scale=2)
plt.xlabel("cool")
plt.grid(True)
plt.legend()
plt.show()
print(pcavals)
print(featureVectorT)

x = np.arange(1,len(pcavalsratio)+1)
plt.figure()
plt.bar(x,pcavalsratio)
plt.show()








#### neural net


xTrain, xTest, yTrain, yTest = train_test_split(featureVector, targets, test_size = 0.1, random_state = 42)
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


