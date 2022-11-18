import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy.polynomial.polynomial as poly


pulsar_file = 'data/HTRU2/HTRU_2.csv'


pulsar_file_df = pd.DataFrame(pd.read_csv(pulsar_file, header=None))
pulsar_file_df.columns = ['mean_ip', 'std_ip', 'kurtosis_ip', 'skewness_ip',
                          'mean_dm', 'std_dm', 'kurtosis_dm', 'skewness_dm',
                          'class']




#print(pulsar_file_df.head(10)[pulsar_file_df["class"]==0])

#print(test[test["class"]==1])

#print(pulsar_file_df.drop(['class'], axis=1).corr())

#plt.figure(figsize=(8,6))
#_= sns.heatmap(pulsar_file_df.drop(['class'], axis=1).corr(), annot=True)
#plt.show()

#plt.figure()
#_ = sns.pairplot(pulsar_file_df, kind='scatter', hue='class', plot_kws=dict(ec='w'))
#plt.show()