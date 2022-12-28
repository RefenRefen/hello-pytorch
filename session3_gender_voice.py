import torch.nn as nn
import pandas as pd
import numpy as np

raw_data = pd.read_csv('./Data/gender_voice_dataset.csv')
raw_data['label'] = raw_data.label.apply(lambda x: 1 if x == 'male' else 0)
data = raw_data.iloc[:, :20]
data = data.values

raw_label = raw_data.iloc[:, 20]
labels = raw_label.values
print(labels)

idx = np.arange(data.shape[0])
np.random.shuffle(idx)
data = data[idx, :]
labels = labels[idx]


