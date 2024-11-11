# setup
import os
import datetime

import IPython
import IPython.display
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import tensorflow as tf

mpl.rcParams['figure.figsize'] = (8, 6)
mpl.rcParams['axes.grid'] = False

# Load dataset
csv_path = 'datasets/us_stock_2020_to_2024.csv'

df=pd.read_csv(csv_path)

# Slice [start:stop:step], starting from index 5 take every 6th record.
#df = df[5::6]

date_time = pd.to_datetime(df.pop('Date Time'), format='%d-%m-%Y')
timestamp_s = date_time.map(pd.Timestamp.timestamp)

#print(df.head())
df = df[['Apple_Price', 'Meta_Price','Tesla_Price','Google_Price','Nvidia_Price','Amazon_Price']]

print("Input data:")
print(df.head())

plot_cols = ['Apple_Price']
plot_features = df[plot_cols]
plot_features.index = date_time
_ = plot_features.plot(subplots=True)

# Statistics of dataset
print(df.describe().transpose())

# Splitting the data 70% training 20% validation 10% testing
column_indices = {name: i for i, name in enumerate(df.columns)}

n = len(df)
train_df = df[0:int(n*0.7)]
val_df = df[int(n*0.7):int(n*0.9)]
test_df = df[int(n*0.9):]

num_features = df.shape[1]

print("Length of dataset: ", n)
print("Number of features: ", num_features)

# Normalize the data
train_mean = train_df.mean()
train_std = train_df.std()

train_df = (train_df - train_mean) / train_std
val_df = (val_df - train_mean) / train_std
test_df = (test_df - train_mean) / train_std

# RNN with LSTM
wide_window = WindowGenerator(
    input_width=6, label_width=1, shift=1,
    label_columns=['Apple_Price'])

lstm_model = tf.keras.models.Sequential([
    # Shape [batch, time, features] => [batch, time, lstm_units]
    tf.keras.layers.LSTM(32, return_sequences=True),
    # Shape => [batch, time, features]
    tf.keras.layers.Dense(units=1)
])

print('Input shape:', wide_window.example[0].shape)
print('Output shape:', lstm_model(wide_window.example[0]).shape)

# Training procedure
MAX_EPOCHS = 20

# Training
val_performance = {}
performance = {}

history = compile_and_fit(lstm_model, wide_window)

val_performance['lstm'] = lstm_model.evaluate(wide_window.val, return_dict=True)
performance['lstm'] = lstm_model.evaluate(wide_window.test, verbose=0, return_dict=True)

wide_window.plot(lstm_model)