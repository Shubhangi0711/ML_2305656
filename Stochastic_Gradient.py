import random


X = [1, 2, 3, 4, 5]
Y = [2, 4, 6, 8, 10]

w = 0.0
b = 0.0
lr = 0.01
epochs = 50

n = len(X)

for _ in range(epochs):
    for i in range(n):
        y_pred = w * X[i] + b
        error = y_pred - Y[i]

        w = w - lr * error * X[i]
        b = b - lr * error

print("Weight:", w)
print("Bias:", b)
