import matplotlib.pyplot as plt
import numpy as np
from sklearn import datasets, linear_model
from sklearn.metrics import mean_squared_error, r2_score

# Load dataset
X, y = datasets.load_diabetes(return_X_y=True)

# For simple regression, use only one feature
X = X[:, np.newaxis, 2]

# Split into train and test sets
X_train, X_test = X[:-20], X[-20:]
y_train, y_test = y[:-20], y[-20:]

# Train model
regr = linear_model.LinearRegression()
regr.fit(X_train, y_train)

# Predict and evaluate
y_pred = regr.predict(X_test)
print("Coefficients:", regr.coef_)
print("Mean squared error: %.2f" % mean_squared_error(y_test, y_pred))
print("R2 score: %.2f" % r2_score(y_test, y_pred))

# Visualize
plt.scatter(X_test, y_test, color="black")
plt.plot(X_test, y_pred, color="blue", linewidth=3)
plt.show()
