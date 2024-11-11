# setup
import os
import datetime

import keras
import IPython
import IPython.display
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import tensorflow as tf

model = keras.models.load_model('rnn_all.keras')

# data for prediction
mpl.rcParams['figure.figsize'] = (8, 6)
mpl.rcParams['axes.grid'] = False

# Load dataset
csv_path = 'datasets/all_stocks.csv'

df=pd.read_csv(csv_path)

# Slice [start:stop:step], starting from index 5 take every 6th record.
df = df[['Date Time', 'Apple_Price','Meta_Price','Tesla_Price','Google_Price','Nvidia_Price','Amazon_Price', 'Microsoft_Price','Netflix_Price']]

# Values for denormalization
df2 = df[['Apple_Price','Meta_Price','Tesla_Price','Google_Price','Nvidia_Price','Amazon_Price', 'Microsoft_Price','Netflix_Price']]
n = len(df2)
train_df = df2[0:int(n*0.7)]
train_mean = train_df.mean()
train_std = train_df.std()
print(train_mean)
print(train_std)

df = df[-51:-1]
#df = df[5::6]

print("Input data:")
print(df)

date_time = pd.to_datetime(df.pop('Date Time'), format='%Y-%m-%d')
timestamp_s = date_time.map(pd.Timestamp.timestamp)

plot_cols = ['Apple_Price', 'Meta_Price','Tesla_Price','Google_Price','Nvidia_Price','Amazon_Price']
plot_features = df[plot_cols]
plot_features.index = date_time
_ = plot_features.plot(subplots=True)
plt.show()

# Statistics of dataset
# print(df.describe().transpose())

# Splitting the data 70% training 20% validation 10% testing
column_indices = {name: i for i, name in enumerate(df.columns)}

n = len(df)
# train_df = df[0:int(n*0.7)]
# val_df = df[int(n*0.7):int(n*0.9)]
# test_df = df[int(n*0.9):]

num_features = df.shape[1]

print("Length of dataset: ", n)
print("Number of features: ", num_features)

# Normalize the data
df = (df - train_mean) / train_std
print(df.head())
data = np.array(df, dtype=np.float32)
ds = tf.keras.utils.timeseries_dataset_from_array(data=data,targets=None,sequence_length=20,sequence_stride=1,shuffle=True,batch_size=32,)

output = model.predict(ds, verbose=0)

# De-normalize
for i in range(0,len(output)):
    for j in range(0,len(output[i])):
        output[i][j] = output[i][j] * train_std + train_mean

print(output[0][-1:])
#print(output)