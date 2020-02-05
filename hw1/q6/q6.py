import random
import numpy as np
e = np.exp(1)
# np.log is ln
# np.log10 is log

def alphaderiv(arr, n, alpha, beta):
    sigma = 0.0
    for i in arr:
        sigma += e ** (-1 * ((i - alpha)/beta))
    result = (n - sigma) / beta
    return result
    
def betaderiv(arr, n, alpha, beta):
    sigma1 = 0.000
    for i in arr:
        sigma1 += (i - alpha)/(beta ** 2)
    sigma2 = 0.000
    for i in arr:
        sigma2 += (((i - alpha)/(beta ** 2)) * (e ** (-1 * (i - alpha)/beta) ) )
    result = sigma1 - sigma2 - (n/beta)
    return result

def log_likelihood_func(arr, n, alpha, beta):
    result = -1 * n * np.log(beta)
    sigma1 = 0.0
    for i in arr:
        sigma1 += (i - alpha) / beta
    sigma2 = 0.0
    for j in arr:
        sigma2 += e ** (-1 * (i - alpha) / beta)
    result = result - sigma1 - sigma2
    return result


def generate_random_arr(n, min=0, max=1):
    arr = []
    for i in range(0, n):
        arr.append(random.uniform(min, max))
    return arr

def calc_step_size(x, learning_rate):
    return x * learning_rate

def run_ml_algorithm(iterations, learning_rate, n, alpha, beta):
    alpha_arr = []
    beta_arr = []
    for j in range(0, 10):

        print("step ", j + 1, "/ 10")
        alpha = 0
        beta = 1    
        dset = generate_random_arr(n, -1, 1)
        for i in range(0, iterations):
            # print("likelihood:", log_likelihood_func(dset, n, alpha, beta))
            alphad = alphaderiv(dset, n, alpha, beta)
            betad = betaderiv(dset, n, alpha, beta)
            alpha_step_size = alphad * learning_rate
            alpha = alpha + alpha_step_size
            
            beta_step_size = betad * learning_rate
            beta = beta + beta_step_size
        print("alpha: ", alpha)
        print("beta: ", beta)
        alpha_arr.append(alpha)
        beta_arr.append(beta)
    print("sd of alpha is: ", np.std(alpha_arr))
    print("mean of alpha is", np.mean(alpha_arr))
    print("sd of beta is: ", np.std(beta_arr))
    print("mean of beta is", np.mean(beta_arr))
    print("---")


iterations = 500
learning_rate = 0.0001
print("Test for n=100")
run_ml_algorithm(iterations, learning_rate, 100, 0, 1)
print("######")
print("Test for n=1000")
run_ml_algorithm(iterations, learning_rate, 1000, 0, 1)