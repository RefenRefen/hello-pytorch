import torch
import torch.nn as nn
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

raw_data = pd.read_csv('./Data/gender_voice_dataset.csv')
raw_data['label'] = raw_data.label.apply(lambda x: 1 if x == 'male' else 0)
data = raw_data.iloc[:, :20]
data = data.values

raw_label = raw_data.iloc[:, 20]
labels = raw_label.values
print(labels)

# idx = np.arange(data.shape[0])
# np.random.shuffle(idx)
# data = data[idx, :]
# labels = labels[idx]

test_ratio = 0.2
valid_ratio = 0.1

x_train, x_test, y_train, y_test = train_test_split(data, labels, test_size=test_ratio, random_state=1, shuffle=True)
x_train, x_valid, y_train, y_valid = train_test_split(x_train, y_train, test_size=valid_ratio, random_state=1, shuffle=True)
print(x_train.shape)

x_train = torch.tensor(x_train).float()
x_test = torch.tensor(x_test).float()
x_valid = torch.tensor(x_valid).float()

y_train = torch.tensor(y_train).float()
y_test = torch.tensor(y_test).float()
y_valid = torch.tensor(y_valid).float()

num_features = 20
num_classes = 1
num_samples_train = x_train.shape[0]
num_samples_valid = x_valid.shape[0]
num_samples_test = x_test.shape[0]

model = nn.Sequential(nn.Linear(num_features, 10),
                      nn.ReLU(),
                      nn.Linear(10, num_classes),
                      nn.Sigmoid())
loss = nn.BCELoss()
optimizer = torch.optim.SGD(model.parameters(), lr=0.001)

num_epochs = 200

for epoch in range(num_epochs):
    optimizer.zero_grad()
    yp = model(x_train)
    y_train = y_train.reshape(-1, 1)
    loss_value = loss(yp, y_train)
    yp = torch.round(yp[:, 0])
    num_corrects = torch.sum(yp == y_train.reshape(1, -1))
    acc_train = num_corrects.float() / float(num_samples_train)
    loss_value.backward()
    optimizer.step()

    yp = model(x_valid)
    num_corrects = torch.sum(yp[:, 0].round() == y_valid)
    acc_valid = num_corrects.float() / float(num_samples_valid)
    print('Epoch:', epoch, 'Train Loss:', loss_value.item(),
          ', Train Accuracy: ', acc_train.item(),
          ', Validation Accuracy: ', acc_valid.item())

yp = model(x_test)
num_corrects = torch.sum(yp[:, 0].round() == y_test)
acc_test = num_corrects.float() / float(num_samples_test)
print('Test Accuracy: ', acc_test.item())



