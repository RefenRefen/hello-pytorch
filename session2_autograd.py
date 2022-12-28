import torch
from sklearn.datasets._samples_generator import make_blobs
from matplotlib import pyplot as plt
from numpy import where
import torch.nn as nn

num_features = 2
num_classes = 3
num_samples = 1000

x, y = make_blobs(n_samples=num_samples, n_features=num_features, centers=num_classes, cluster_std=1.2, random_state=3)
for class_value in range(num_classes):
    row_idx = where(y == class_value)
    plt.scatter(x[row_idx, 0], x[row_idx, 1])
# plt.scatter(x[:, 0], x[:, 1], c=y)
plt.show()
x_train = torch.tensor(x[:int(0.6*num_samples), :]).float()
y_train = torch.tensor(y[:int(0.6*num_samples)]).long()
x_valid = torch.tensor(x[int(0.6*num_samples): int(0.75*num_samples), :]).float()
y_valid = torch.tensor(y[int(0.6*num_samples): int(0.75*num_samples)]).long()
x_test = torch.tensor(x[int(0.75*num_samples):, :]).float()
y_test = torch.tensor(y[int(0.75*num_samples):]).long()
#
# x = torch.tensor(1.0, requires_grad=True)
#
# y = x**2
#
# y.backward()
# print(x.grad)
#
# w = torch.tensor(3.0, requires_grad=True)
# b = torch.tensor(10.0, requires_grad=True)
# y2 = w * x + b
# y2.backward()
# print(w.grad, x.grad, b.grad)

# x = torch.rand(10, 3)
# y = torch.rand(10, 2)

model = nn.Sequential(nn.Linear(num_features, 10),
                      nn.ReLU(),
                      nn.Linear(10, num_classes))
# model = nn.Linear(num_features, num_classes)
# loss_function = torch.nn.MSELoss()
loss_function = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(model.parameters(), lr=0.001)

num_epochs = 200
x = torch.FloatTensor(x)
# y = torch.tensor(y, dtype=torch.int64)
# y = torch.tensor(y).long()
y = torch.LongTensor(y)
for epoch in range(num_epochs):
    optimizer.zero_grad()
    yp = model(x_train)
    loss_value = loss_function(yp, y_train)
    num_corrects = torch.sum(torch.max(yp, 1)[1] == y_train)
    acc_train = num_corrects.float()/num_samples.__float__()
    loss_value.backward()
    optimizer.step()

    yp = model(x_valid)
    num_corrects = torch.sum(torch.max(yp, 1)[1] == y_valid)
    acc_valid = num_corrects.float() / num_samples.__float__()

    print('Epoch: ', epoch, 'Train Loss: ', loss_value.item(),
          "Train Accuracy: ", acc_train.item(),
          "Validation Accuracy: ", acc_valid.item())

yp = model(x_test)
num_corrects = torch.sum(torch.max(yp, 1)[1] == y_test)
acc_test = num_corrects.float() / num_samples.__float__()

print("Test Accuracy: ", acc_test.item())
