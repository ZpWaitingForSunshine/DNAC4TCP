import time

import ray
import numpy as np
import sys

from utils.nonlocal_function import find_min_indices, knn2
from utils.tools import calEuclidean, cgsolve2

# Y_ref 是Y的内存对象
@ray.remote(num_cpus=20)
def knn(indices, rows, cols, Pstepsize, index, Y):
    # print('knn+ =')
    indices_ = indices.copy().astype('float64')
    # extract thte patches
    # get the vectoer
    # Y = ray.get([Y_ref])
    row = int(index % rows)
    col = int((index - row) / rows)
    print(row, col, index)
    patch = Y[row: row + Pstepsize, col: col + Pstepsize, :]
    nn = patch.shape
    vec = np.reshape(patch, [nn[0] * nn[1] * nn[2]], order='F')

    for i in range(indices.shape[1]):
        row = int(indices[0][i] % rows)
        col = int((indices[0][i] - row) / rows)
        patch = Y[row: row + Pstepsize, col: col + Pstepsize, :]
        nn = patch.shape
        cur_vector = np.reshape(patch, [nn[0] * nn[1] * nn[2]], order='F')
        # print(cur_vector.shape)
        # print(vec.shape)
        distance = calEuclidean(cur_vector, vec)
        # print(distance.shape)
        indices_[1][i] = distance
    # 找到第三列不是-1的元素的索引
    # column_indices = np.where(indices[2, :] == -1)[0]
    # print(indices_[1])
    return indices_

@ray.remote(num_cpus=10)
def partitions_group(indices, rows, cols, patsize, Y, PN):
    indices_ = indices.copy()
    indices_[1] = sys.maxsize
    num = int(np.ceil(indices.shape[1] / PN))
    indices_set = []
    for i in range(num - 1):
        indices_ = knn2(indices_, rows, cols, patsize, indices_[0][0], Y)
        min_indices = find_min_indices(indices_, PN)
        indices_set.append(indices_[:, min_indices][0])
        indices_ = np.delete(indices_, min_indices, axis=1)
    indices_set.append(indices_[0])
    return indices_set

@ray.remote(num_cpus=10)
def cg(indices, rr, mu, rate, s):
    print("分区开始运行CG")
    t1 = time.time()
    # maxtrix = []
    # for rr in rr_list:
    #     maxtrix.append(rr.T)
    rr_matrix = rr[:, :, indices[0]: indices[-1] + 1]
    # rr_matrix = np.stack(maxtrix, axis=2)
    nn = rr_matrix.shape
    rr = np.reshape(rr_matrix, [nn[0] * nn[1] * nn[2]], order='F')
    res = cgsolve2(rr, nn, mu, rate, s)

    t2 = time.time()
    print("分区更新完成，用时%d秒" % (t2 - t1))
    return res




