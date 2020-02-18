import random
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import math
e = np.exp(1)


def cost(X, Y, W, n, d):
    cst = 0
    for i in range(0, n):    
        tmp = 0
        iner_sigma = 0
        for j in range(0, d+1):
            iner_sigma += W[j] * X[i][j]
        tmp = iner_sigma - Y[i]
        cst += tmp * tmp
    return cst
    

def partial_deriv(X, Y, W, n, d, k):
    res = 0
    for i in range(0, n):
        tmp = 0
        iner_sigma = 0
        for j in range(0, d+1):
            iner_sigma += W[j] * X[i][j]
        tmp = (iner_sigma - Y[i]) * X[i][k]
        res += tmp
    return 2 * res


def gradient_descent(X, Y, W, n, d, step_size, num_steps):
    cst = np.zeros(num_steps)
    for i in range (0, num_steps):
        old_W = W
        for j in range (0, d+1):
            pd = partial_deriv(X, Y, W, n, d, j)
            W[j] = old_W[j] - step_size * partial_deriv(X, Y, W, n, d, j)
        cst[i] = cost(X, Y, W, n, d)
    return W, cst


def calculate_dataset_values(dset):
    # Normalize data
    dset = (dset - dset.mean())/dset.std()
    n = len(dset)
    d = len(dset.columns) - 1
    print(dset.head())

    # Initialize array of w with all 0
    W = np.zeros(d+1)
    # Add 1s column to X
    X = dset.iloc[:,0:d]
    ones = np.ones([X.shape[0],1])
    X = np.concatenate((ones,X),axis=1)
    # Generaet Y Matrix
    Y = dset.iloc[:,d].values
    # Step Size and Number of Steps
    return X, Y, W, n, d
    

def load_d1():
    # Load dataset
    dset = pd.read_csv('./dset_home.csv')
    step_size = 0.0001
    num_steps = 1000
    X, Y, W, n, d = calculate_dataset_values(dset)
    return X, Y, W, n, d, step_size, num_steps

def load_d2():
    # Load dataset
    dset = pd.read_csv('./housing.csv')
    step_size = 0.0001
    num_steps = 150
    X, Y, W, n, d = calculate_dataset_values(dset)
    return X, Y, W, n, d, step_size, num_steps

# Uncomment Each line to load and run code for each dataset!
X, Y, W, n, d, step_size, num_steps = load_d1()
# X, Y, W, n, d, step_size, num_steps = load_d2()
# X, Y, W, n, d, step_size, num_steps = load_d3()
# X, Y, W, n, d, step_size, num_steps = load_d4()
# X, Y, W, n, d, step_size, num_steps = load_d5()

print('First W:', W)
W_final, cst = gradient_descent(X, Y, W, n, d, step_size, num_steps)
print('Final W:', W_final)

fig, ax = plt.subplots()
ax.plot(np.arange(num_steps), cst, 'r')
ax.set_xlabel('Iterations')
ax.set_ylabel('Error')
ax.set_title('Error / Iter')
plt.show()