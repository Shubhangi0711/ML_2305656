import numpy as np

X = np.array([1, 2, 3, 4,5])
Y = np.array([6,3,4,4,6])   # y = 2x + 1

m = 0
c = 0

lr = 0.01
epochs = 1000

n = len(X)

for _ in range(epochs):
    y_pred = m*X + c

    dm = (-2/n) * sum(X * (Y - y_pred))
    dc = (-2/n) * sum(Y - y_pred)

    m = m - lr * dm
    c = c - lr * dc

print("m =", m)
print("c =", c)
