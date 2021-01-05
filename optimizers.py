# -*- coding: utf-8 -*-
"""
Created on Mon Dec 28 14:13:22 2020

@author: leonardoqueiroz
"""

''' OPTIMIZERS '''

import numpy as np


X_train, Y_train, X_test, Y_test = None, None, None, None
learning_rate = None


# Functions
def loss_fn(X_train, Y_train, W):
    pass
    
def initialize_weights():
    pass

def compute_gradient(loss_fn, data, W):
    pass

''' Random Search '''
# Assume X_train is the data where each column is an example (e.g. 3073 x 50000)
# Assume Y_train are the labels (e.g. 1D array of 50000)
# Assume the function L evaluates the loss function

bestloss = float("inf") # Python assigns the highest possible float value
for num in range(1000):
    W = np.random.randn(10, 3073) * 0.0001 # Generate random parameters
    loss = loss_fn(X_train, Y_train, W) # Get losso over the entire training set
    if loss < bestloss: # Keep track of the best solution
        bestloss = loss
        bestW = W
    print('In attempt %d the loss was %f, best %f' % (num, loss, bestloss))

# Assume X_test is [3073 x 10000], Y_test [10000 x 1]    
scores = bestW.dot(X_test) # 10 x 10000, the class scores for all test examples
# Find the index with max score in each column (the predicted class)
Y_test_predicted = np.argmax(scores, axis = 0)
# Calculate accuracy (fraction of predictions that are correct)
np.mean(Y_test_predicted == Y_test)
# returns 0.1555   
    
    
''' Gradient Descent ''' 
# Vanilla gradient descent
num_steps = None
data = None
W = initialize_weights()
for t in range(num_steps):
    dw = compute_gradient(W)
    W -= learning_rate * dw
  

''' Stochastic Gradient Descente - SGD '''
def sample_data(data, batch_size):
    pass
data = None
batch_size = None

W = initialize_weights()
for t in range(num_steps):
    minibatch = sample_data(data, batch_size)
    dw = compute_gradient(loss_fn, minibatch, W)
    W -= learning_rate * dw

    
''' Stochastic Gradient Descente - SGD  + Momentum '''
v = 0
rho = None # 0.9 or 0.99
# Version 1
for t in range(num_steps):
    dw = compute_gradient(W)
    v = rho * v + dw
    W -= learning_rate * v     

# Version 2
for t in range(num_steps):
    dw = compute_gradient(W)
    v = rho * v - learning_rate * dw
    W += v


''' AdaGrad '''
     
grad_squared = 0
for t in range(num_steps):
    dw = compute_gradient(W)
    grad_squared += dw * dw
    W -= learning_rate * dw / (grad_squared.sqrt() + 1e-7)


''' RMSProp '''
grad_squared = 0
decay_rate = None

for t in range(num_steps):
    dw = compute_gradient(W)
    grad_squared = decay_rate * grad_squared + (1 - decay_rate) * dw * dw
    W -= learning_rate * dw / (grad_squared.sqrt() + 1e-7)
    
    
''' Adam: RMSProp + Momentum '''
moment1 = 0 # Analogous to the velocity on SGD + Momentum
moment2 = 0 # Analogous to the Leaky exponential average of the squared gradients on RMSProp
beta1, beta2 = None, None
for t in range(num_steps):
    dw = compute_gradient(W)
    moment1 = beta1 * moment1 + (1 - beta1) * dw
    moment2 = beta2 * moment2 + (1 - beta2) * dw * dw
    W -= learning_rate * moment1 / (moment2.sqrt() + 1e-7)
    
''' Adam: Bias correction '''
# Bias correction for the fact that first and second moment estimates start at zero
moment1 = 0 # Analogous to the velocity on SGD + Momentum
moment2 = 0 # Analogous to the Leaky exponential average of the squared gradients on RMSProp
beta1, beta2 = None, None # 0.9, 0.999
learning_rate = None # 1e-3 , 5e-4 , 1e-4 
for t in range(num_steps):
    dw = compute_gradient(W)
    moment1 = beta1 * moment1 + (1 - beta1) * dw
    moment2 = beta2 * moment2 + (1 - beta2) * dw * dw
    moment1_unbias = moment1 / (1 - beta1 ** t)
    moment2_unbias = moment2 / (1 - beta2 ** t)
    W -= learning_rate * moment1_unbias / (moment2_unbias.sqrt() + 1e-7)
    



























