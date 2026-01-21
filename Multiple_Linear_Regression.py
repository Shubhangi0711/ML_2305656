import numpy as np
from sklearn.linear_model import LinearRegression

x = np.array([[3,8],[4,5],[5,7],[6,3],[2,1]])
y = np.array([-3.7,3.5,2.5,11.5,5.7])

model = LinearRegression()
model.fit(x, y)

print(model.coef_)
print(model.intercept_)
print(model.predict([[3,2]]))
