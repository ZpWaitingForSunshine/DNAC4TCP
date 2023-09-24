import numpy as np
from scipy.ndimage import convolve
from scipy import signal

def gaussian(X, s):
    X_copy = X.copy()
    for i in range(X.shape[2]):
        X_copy[:, :, i] = convolve(X[:, :, i], s, mode='reflect')
    return X_copy

# 下采样 downsample
def downsample(inMatrix, rate):
    inMatrix_ = inMatrix.copy()
    return inMatrix_[::rate, 0::rate, :]

def upsample(I_Interpolated, ratio):
    L = 45
    [r, c, b] = I_Interpolated.shape
    kernel = ratio * signal.firwin(L, 1 / ratio)
    kernel = np.reshape(kernel, [1, len(kernel)])
    I1LRU = np.zeros([ratio * r, ratio * c, b])
    I1LRU[0::ratio, 0::ratio, :] = I_Interpolated.copy()
    for ii in range(b):
        t = I1LRU[:, :, ii]
        t = convolve(t.T, kernel, mode='wrap')
        I1LRU[:, :, ii] = convolve(t.T, kernel, mode='wrap')
    return I1LRU

def calEuclidean(x, y):
    dist = np.sqrt(np.sum(np.square(x-y)))   # 注意：np.array 类型的数据可以直接进行向量、矩阵加减运算。np.square 是对每个元素求平均~~~~
    # print(dist)
    return dist

def cgsolve2(b, nn, mu, rate, s):
    # b = np.reshape(b, b.shape[0], order='F')
    n = len(b)
    maxiters = 50
    normb = np.linalg.norm(b)
    x = np.zeros(n)
    r = b.copy()
    rtr = np.dot(r.T, r)
    d = r
    niters = 0
    while np.sqrt(rtr) / normb > 2e-6 and niters < maxiters:
        niters = niters + 1
        Ad = myfun2(d, mu, rate, nn, s)
        alpha = rtr / np.dot(d.T, Ad)
        x = x + alpha * d
        r = r - alpha * Ad
        rtrold = rtr
        rtr = np.dot(r, r)
        beta = rtr / rtrold
        d = r + beta * d

    return np.reshape(x, nn, order='F')


def myfun2(X, mu, rate, nn, s):
    X_ = np.reshape(X, nn, order='F')
    ours = gaussian(X_, s)
    ours = downsample(ours, rate)
    ours = upsample(ours, rate)
    ours = gaussian(ours, s)
    re = 2 * ours + mu * X_
    return np.reshape(re, re.shape[0] * re.shape[1] * re.shape[2], order='F')


