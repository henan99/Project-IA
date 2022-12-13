import numpy as np
import pandas as pd
from keras.models import Sequential
from keras.layers import Dense
import matplotlib.pyplot as plt
import numpy.polynomial.polynomial as poly
from sklearn.preprocessing import StandardScaler
import preprocessing as prep
from sklearn.model_selection import train_test_split





pulsar_file = 'data/HTRU2/HTRU_2.csv'




X_train, X_test, y_train, y_test = prep.X_train, prep.X_test, prep.y_train, prep.y_test
print(np.shape(np.array(X_train)))

# defining model
simpleAnn = Sequential()
   
simpleAnn.add(Dense(32, activation = 'relu'))
epochs = 10
batch_size=32
#simpleAnn.Dropout(0.1),
simpleAnn.compile(loss='binary_crossentropy', optimizer='sgd',metrics=["Precision","Recall"])
history = simpleAnn.fit(X_train, y_train, epochs=epochs, batch_size=batch_size)

y_pred = simpleAnn.evaluate(X_test, y_test)
y_pred2 = simpleAnn.evaluate(X_train, y_train)

print("the score for test is: ")
print(y_pred)


print("the score for train is: ")
print(y_pred2)
print(simpleAnn.metrics_names)


plt.figure(figsize=(12,6))
plt.subplot(1,2,1)
plt.plot(np.arange(epochs), history.history['loss'])
plt.plot(np.arange(epochs), history.history['val_loss'])
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.xlim(0,epochs)
#plt.ylim(0,0.25)
plt.grid()
plt.subplot(1,2,2)
plt.plot(np.arange(epochs), history.history['accuracy'])
plt.plot(np.arange(epochs), history.history['val_accuracy'])
plt.xlabel("Epoch")
plt.ylabel("Precision")
plt.xlim(0,epochs)
#plt.ylim(0.9,1)
plt.grid()