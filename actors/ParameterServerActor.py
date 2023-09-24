import time

import ray
import numpy as np
from scipy.sparse import csr_matrix

@ray.remote
class ParameterServerActor(object):
    def __init__(self, nn):
        self.HR_HSI = np.zeros(nn)
        self.nn = nn

    def put_HR_HSI(self, *HSI_Piece):
        # print(HSI_Piece)
        self.HR_HSI = np.zeros(self.HR_HSI.shape)
        t1 = time.time()
        for hsi in HSI_Piece:
            temp = np.zeros(self.HR_HSI.shape)
            print(self.HR_HSI.shape)
            for i in range(self.HR_HSI.shape[2]):
                # print(hsi.data[i])
                # print(hsi.col_indices)
                # print(hsi.row_offset)
                temp[:, :, i] = csr_matrix((hsi.data[i], hsi.col_indices,
                                      hsi.row_offset), shape=(self.nn[0], self.nn[1])).toarray()
            print("我在更新", time.ctime())
            self.HR_HSI += temp
        t2 = time.time()
        print("合并花了", t2 - t1, " s")
        print("尺寸", self.HR_HSI.shape)
        return self.HR_HSI
        # self.net.apply_gradients(np.mean(gradients, axis=0))
        # return self.net.variables.get_flat()

    def get_HR_HSI(self):
        return self.HR_HSI

    def reset(self):
        self.HR_HSI = np.zeros(self.HR_HSI.shape)


