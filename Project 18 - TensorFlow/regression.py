import tensorflow as tf
import pandas as pd
import datetime
import matplotlib.pyplot as plt
import numpy as np
from sklearn import preprocessing


features = pd.read_csv('temps.csv')

years = features['year']
months = features['month']
days = features['day']

dates = [str(int(year)) + '-' + str(int(month)) + '-' + str(int(day)) for year, month, day in zip(years, months, days)]
dates = [datetime.datetime.strptime(date, '%Y-%m-%d') for date in dates]

plt.style.use('fivethirtyeight')

fig, axes = plt.subplots(nrows=2, ncols=2, figsize=(10, 10))
fig.autofmt_xdate(rotation=45)

axes[0][0].plot(dates, features['actual'])
axes[0][0].set_xlabel('')
axes[0][0].set_ylabel('Temperature')
axes[0][0].set_title('Max Temp')

axes[0][1].plot(dates, features['temp_1'])
axes[0][1].set_xlabel('')
axes[0][1].set_ylabel('Temperature')
axes[0][1].set_title('Previous Max Temp')

axes[1][0].plot(dates, features['temp_2'])
axes[1][0].set_xlabel('')
axes[1][0].set_ylabel('Temperature')
axes[1][0].set_title('Two Days Prior Max Temp')

axes[1][1].plot(dates, features['friend'])
axes[1][1].set_xlabel('')
axes[1][1].set_ylabel('Temperature')
axes[1][1].set_title('Friend Estimate')

plt.tight_layout(pad=2)
plt.show()

features = pd.get_dummies(features)

labels = np.array(features['actual'])
features = features.drop('actual', axis=1)

features_list = list(features.columns)

features = np.array(features)

input_features = preprocessing.StandardScaler().fit_transform(features)

model = tf.keras.Sequential()
model.add(tf.keras.layers.Dense(16, kernel_initializer='random_normal', kernel_regularizer=tf.keras.regularizers.l2(0.03)))
model.add(tf.keras.layers.Dense(32, kernel_initializer='random_normal', kernel_regularizer=tf.keras.regularizers.l2(0.03)))
model.add(tf.keras.layers.Dense(1, kernel_initializer='random_normal', kernel_regularizer=tf.keras.regularizers.l2(0.03)))

model.compile(optimizer=tf.keras.optimizers.SGD(0.001), loss='mean_squared_error')
model.fit(input_features, labels, validation_split=0.25, epochs=100, batch_size=256)

print(model.summary())

predict = model.predict(input_features)

true_data = pd.DataFrame(data={'date': dates, 'actual': labels})
predicted_data = pd.DataFrame(data={'date': dates, 'predicted': predict.reshape(-1)})

plt.plot(true_data['date'], true_data['actual'], 'b-', label='actual')
plt.plot(predicted_data['date'], predicted_data['predicted'], 'ro', label='predicted')

plt.xticks(rotation='60')
plt.legend()
plt.xlabel('Date')
plt.ylabel('Max Temp')
plt.title('Actual and Predicted Values')

plt.show()

