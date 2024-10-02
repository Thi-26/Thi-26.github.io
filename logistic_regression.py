

import numpy as np
import matplotlib.pyplot as plt
#data
X = np.array([0.245,0.247,0.285,0.299,0.327,0.347,0.356,
0.36,0.363,0.364,0.398,0.4,0.409,0.421,
0.432,0.473,0.509,0.529,0.561,0.569,0.594,
0.638,0.656,0.816,0.853,0.938,1.036,1.045])
Y = np.array([0,0,1,1,1,1,0,1,0,1,0,1,0,1,0,1,1,1,0,0,1,
1,1,1,1,1,1,1])
#initialize theta parameters and learning rate
np.random.seed(0)
theta0 = random.rand()
theta1 = random.rand()
learning_rate = 1*e-4
interations = 100

#logisstic_function
def logistic_function(z):
  return 1  /  (1+ np.exp(-z))

#precdiction function
def predict (X , theta0, theta1):
  z = theta0 + theta1 * X
  gz = logistic_function(z)
  return gz

#cost_function
def cost_function(X, Y_true, theta0, theta1):
  m = len(X)
  epsilon = 1*e - 15
  Y_pred = predict(X, theta0, theta1)
  cost = - (1 / m)  * np.sum(Y_true *  np.log(Y_pred + epsilon) + (1- Y_true) * np.log( 1- Y_pred))
  return cost
  #gradient descent
def gradient_descent(X, Y, theta0, theta1, learning_rate):
    m = len(X)
    gradient0 = (1 / m)  *  np.sum(predict (X, theta0, theta1) - Y)
    gradient1 = (1 / m) * np.sum((predict (X, theta0, theta1) - Y) * X )
    new_theta0 = theta0 - learning_rate * gradient0
    new_theta1 = theta1 - learning_rate * gradient1
    return new_theta0, new_theta1
for i in range(interations):
    theta0, theta1 = gradient_descent(X, Y, theta0, theta1, learning_rate)

print(theta0, theta1, cost(X, Y_true, theta0, theta1))
