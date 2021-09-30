import numpy as np 
import matplotlib.pyplot as plt 
import seaborn as sns
from scipy.io import loadmat

sns.set(style="ticks", context="paper", palette="pastel")
plt.style.use('seaborn')

data = np.loadtxt('cluster.dat')

plt.scatter(data[0,:], data[1,:])
plt.show()

np.random.seed(1)
data_test = data[:,:20]
m=3
max_iter=2

n_features = data_test.shape[0]
n_data = data_test.shape[1]

w = np.ones((n_features, m))
w = np.multiply(w.T, np.mean(data_test, axis=1)).T + np.random.normal(scale=1e-3, size=w.shape)
m_arr = np.zeros(n_data)

for i in range(max_iter):
    for j in range(n_data):
        e_dst = w - data_test[:,[j]]
        e_dst = np.linalg.norm(e_dst, axis=0)
        arg_dst = np.argmax(e_dst)
        m_arr[j] = arg_dst
    for j in range(m):
        idx = m_arr == j
        print(idx)
        m_data = data_test[:,idx]
        if m_data.shape[0] != 0:
            w[:,j] = np.mean(m_data, axis=1)
    print(w)