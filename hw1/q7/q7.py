import random
import numpy as np
import math
e = np.exp(1)
pi = math.pi 

def p1(x, alpha, beta):
    return (1 / beta) * ( e ** (-1 * (x-alpha) / beta) ) * (e ** (-1 * e ** (-1 * (x-alpha) / beta) ) )

def p2(x, sigma, mu):
    return (1 / (math.sqrt(2 * pi * (sigma ** 2)))) * (e ** (-1 * ((x-mu)**2) / (2 * (sigma ** 2)) ) )

def likelihood(dset, w1, w2, alpha, beta, sigma, mu):
    ll = 1
    for i in dset):
        aa = w1 * p1(i, alpha, beta)
        bb = w2 * p2(i, sigma, mu)
        cc = aa + bb
        ll *= cc
    return ll

def pyi1(x, w1, w2, alpha, beta, sigma, mu):
    aa = w1 * p1(x, alpha, beta)
    bb = w1 * p1(x, alpha, beta) + w2 * p2(x, beta, mu)
    return aa / bb

def pyi2(x, w1, w2, alpha, beta, sigma, mu):
    aa = w2 * p2(x, beta, mu)
    bb = w1 * p1(x, alpha, beta) + w2 * p2(x, beta, mu)
    return aa / bb

def stepc(n, pyi_all, dset):
    res = 0
    for i in range(0, n):
        res += dset[i] * pyi_all[i]
    return np.sum(pyi_all) / res


alpha0 = 
beta0 = 
sigma0 = 
mu0 = 

w10 = 0.6
w20 = 0.4

iterations = 100
t = 0
for iter in range(0, iterations):
    py1_all = []
    py2_all = []
    # step a
    for i in dset:
        py1 = pyi1(i, w10, w20, alpha, beta, sigma, mu)
        py2 = pyi2(i, w10, w20, alpha, beta, sigma, mu)
        py1_all.append(py1)
        py2_all.append(py2)
    
    # step b
    w1_new = np.sum(py1_all) / n
    w2_new = np.sum(py2_all) / n

    # step c
    alpha_new = stepc(n, py1_all, dset)
    sigma_new = stepc(n, py2_all, dset)
