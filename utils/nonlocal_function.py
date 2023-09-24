import numpy as np

from utils.tensor_function import tensor_mode_unfolding
from utils.tools import calEuclidean


def Im2Patch3D(Video, patsize:int, step:int):
    TotalPatNum = int((np.floor((Video.shape[0] - patsize) / step) + 1) * \
                  (np.floor((Video.shape[1] - patsize) / step) + 1))
    Y = np.zeros((int(patsize * patsize), Video.shape[2], TotalPatNum))
    k = 0
    for i in range(patsize):
        for j in range(patsize):
            tempPatch = Video[i: Video.shape[0] - patsize + i + 1: step, j: Video.shape[1] - patsize + j + 1: step, :]
            Y[k, :, :] = tensor_mode_unfolding(tempPatch, mode=2)
            k = k + 1
    return Y


# arr is an 1xN array
def split_average(arr, PN, num):
    # arr = np.arange(121)
    # PN = 11
    # num = 4

    # the number of Patch group
    num_groups = np.ceil(len(arr) / PN)
    # each partition have num_partitions patch
    num_partitions = int(np.ceil(num_groups / num))
    indices_spilt = []
    for i in range(num):
        start = int(i * PN * num_partitions)
        end = int((i + 1) * PN * num_partitions)
        # print(start, end)
        indices_spilt.append(arr[start: end])
    # print(num_partitions)
    return indices_spilt


def find_min_indices(arr, N):
    # 获取第二行的数据
    row = arr[1]
    # 使用argsort函数对第二行进行排序，并获取排序后的索引
    sorted_indices = np.argsort(row)
    # 取前N个最小值的索引
    min_indices = sorted_indices[:N]
    return min_indices


def knn2(indices, rows, cols, Pstepsize, index, Y):
    indices_ = indices.copy().astype('float64')
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
        distance = calEuclidean(cur_vector, vec)
        indices_[1][i] = distance

    return indices_


def indices2Patch(img, indices, Pstepsize, rows, cols):

    Patch = np.zeros((Pstepsize, Pstepsize, img.shape[2], len(indices)))
    for i in range(len(indices)):
        row = int(indices[i] % rows)
        col = int((indices[i] - row) / rows)
        patch = img[row: row + Pstepsize, col: col + Pstepsize, :]
        cube = np.transpose(patch, [1, 0, 2])
        Patch[:, :, :, i] = cube
    return Patch


def getW_Imge_Matrix(nn, rows, cols, patsize):
    W_Img = np.zeros([nn[0], nn[1]])
    for index in range(int(rows * cols)):
        row = int(index % rows)
        col = int((index - row) / rows)
        W_Img[row: patsize + row, col: patsize + col] = \
            W_Img[row: patsize + row, col: patsize + col] + np.ones((patsize, patsize))
    return W_Img

