import random
import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial import distance


def calc_dist(dset):
    res = []
    for i in range(0, n):
        for j in range(i + 1, n):
            d = distance.euclidean(dset[i], dset[j])
            res.append(d)
    return res

def rkf(dset):
    all_distances = calc_dist(dset)
    print("Calculated All Distances")
    dmax = max(all_distances)
    dmin = min(all_distances)
    val = (dmax - dmin) / dmin
    return np.log10(val)

def gen_dataset(n, k):
    dset = []
    rk_list = []
    for i in range(0, n):
        arr = []
        for j in range(0, k):
            x = random.uniform(0, 1)
            arr.append(x)  
        dset.append(arr)
    return dset


k = 100
n = 100
rk1 = []
for i in range(1, k + 1):
    dset = gen_dataset(n, i)
    print("dataset generated with n:", n, "and dim:", i)
    rk = rkf(dset)
    rk1.append(rk)
    print("rk:", rk)

plt.plot(range(1, k+1), rk1, 'o', color='red', )
# plt.xlabel('K')
# plt.ylabel('R(K)')
# plt.title('n=100')
# plt.show(rk1)


k = 100
n = 1000
rk2 = []
for i in range(1, k + 1):
    dset = gen_dataset(n, i)
    print("dataset generated with n:", n, "and dim:", i)
    rk = rkf(dset)
    rk2.append(rk)
    print("rk:", rk)

plt.plot(range(1, k+1), rk2, 'o', color='blue', )
plt.xlabel('K')
plt.ylabel('R(K)')
plt.title('n=1000 is blue. n=100 is red')
plt.show(rk2)