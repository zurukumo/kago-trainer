import csv

import numpy as np
from tensorflow.keras import layers, models

# 変数
file_name = 'reward'
epoch = 200
batch_size = 256

model = models.Sequential([
    layers.Dense(256, activation='relu'),
    layers.Dropout(0.5),
    layers.Dense(512, activation='relu'),
    layers.Dropout(0.5),
    layers.Dense(1),
])

model.compile(optimizer='adam',
              loss='mse',
              metrics=['accuracy', 'mae', 'mse'])

# データ準備
x = []
t = []
with open(file_name + '.csv') as f:
    reader = csv.reader(f, delimiter=",")
    for row in reader:
        row = list(map(int, row))
        tt, row = row[0], row[4:]
        xx = [0] * (7 * 120)
        for h, w in enumerate(row):
            # print(h, w)
            xx[h * 100 + w] = 1

        x.append(xx)
        t.append(tt)

x = np.array(x, np.float32)
t = np.array(t, np.int32)

print(x.shape)
print(x[0])
print(t)
print()
print()
print()

model.fit(x, t, epochs=epoch)
model.save('{}.h5'.format(file_name))
