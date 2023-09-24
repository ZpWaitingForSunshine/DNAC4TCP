
import ray
import numpy as np
import time
import scipy.linalg as sl

import logging


from utils.nonlocal_function import indices2Patch
from classes.Classes import Patch, Factor, SparseTensor
from utils.tensor_function import matricize, ktensor

from scipy.sparse import csr_matrix

@ray.remote(num_cpus=1)
class FactorActor:
    def __init__(self, k1, Y, patsize, rows, cols, nn):
        print("init factors")
        self.parDatalist = []
        # print(len(k1))
        patchList = [] #
        self.E_Img = np.zeros(nn)

        for i in range(len(k1)):
            patch = Patch(k1[i].astype(int), 80, 0)
            patchList.append(patch)

        for patch in patchList:
            ind = patch.Indices

            Ytt1 = indices2Patch(Y, ind, patsize, rows, cols)
            patch.addY2(np.linalg.norm(Ytt1))

            k = patch.Rank

            U1 = np.random.random([patsize, k])
            diag_matrix = np.diag(np.reciprocal(np.sqrt(np.sum(U1 ** 2, axis=0))))
            U1 = np.dot(U1, diag_matrix)

            U2 = np.random.random([patsize, k])
            diag_matrix = np.diag(np.reciprocal(np.sqrt(np.sum(U2 ** 2, axis=0))))
            U2 = np.dot(U2, diag_matrix)

            U3 = np.random.random([nn[2], k])
            diag_matrix = np.diag(np.reciprocal(np.sqrt(np.sum(U3 ** 2, axis=0))))
            U3 = np.dot(U3, diag_matrix)

            U4 = np.random.random([len(ind), k])
            diag_matrix = np.diag(np.reciprocal(np.sqrt(np.sum(U4 ** 2, axis=0))))
            U4 = np.dot(U4, diag_matrix)


            factor = Factor(U1, U2, U3, U4)
            patch.addFactor(factor)
            patch.addLast(10000)
            self.parDatalist.append(patch)

    def getparDatalist(self):
        return self.parDatalist


    def updateFactors(self, X, patsize, rows, cols, M2, Y, lda, mu, R, nn):
        E_Img = np.zeros(nn)
        time_start = time.time()
        patchlist = self.parDatalist
        print("开始更新因子矩阵")
        curPatchlist = []
        for patch in patchlist:
            ind = patch.Indices
            # print(ind)
            # print(X.shape)
            tt1 = indices2Patch(X, ind, patsize, rows, cols)
            SO1 = matricize(tt1)

            Ytt1 = indices2Patch(Y, ind, patsize, rows, cols)
            YSO1 = matricize(Ytt1)

            Mtt1 = indices2Patch(M2, ind, patsize, rows, cols)
            MM2 = matricize(Mtt1)

            indices = patch.Indices

            # 新更新
            # U1
            W1 = sl.khatri_rao(patch.factor.U4, patch.factor.U3)
            W1 = sl.khatri_rao(W1, patch.factor.U2)
            P1 = sl.khatri_rao(patch.factor.U4, np.dot(R, patch.factor.U3))
            P1 = sl.khatri_rao(P1, patch.factor.U2)

            G1 = np.dot(patch.factor.U4.T, patch.factor.U4) * np.dot(patch.factor.U3.T, patch.factor.U3) \
                 * np.dot(patch.factor.U2.T, patch.factor.U2)
            PTP = np.dot(patch.factor.U4.T, patch.factor.U4) * np.dot(np.dot(R, patch.factor.U3).T,
                                                                      np.dot(R, patch.factor.U3)) \
                  * np.dot(patch.factor.U2.T, patch.factor.U2)
            t1 = mu * G1 + 2 * lda * PTP
            # np.dot(patch.M2[0].T, W1)
            patch.factor.U1 = np.dot(np.dot(MM2[0].T, W1) + 2 * lda * np.dot(YSO1[0].T, P1) +
                                     mu * np.dot(SO1[0].T, W1), np.linalg.inv(t1))

            # U2
            W2 = sl.khatri_rao(patch.factor.U4, patch.factor.U3)
            W2 = sl.khatri_rao(W2, patch.factor.U1)
            P2 = sl.khatri_rao(patch.factor.U4, np.dot(R, patch.factor.U3))
            P2 = sl.khatri_rao(P2, patch.factor.U1)
            G2 = np.dot(patch.factor.U4.T, patch.factor.U4) * np.dot(patch.factor.U3.T, patch.factor.U3) \
                 * np.dot(patch.factor.U1.T, patch.factor.U1)
            P2TP2 = np.dot(patch.factor.U4.T, patch.factor.U4) * np.dot(np.dot(R, patch.factor.U3).T,
                                                                        np.dot(R, patch.factor.U3)) \
                    * np.dot(patch.factor.U1.T, patch.factor.U1)
            t2 = mu * G2 + 2 * lda * P2TP2
            patch.factor.U2 = np.dot(np.dot(MM2[1].T, W2) + 2 * lda * np.dot(YSO1[1].T, P2) + mu * np.dot(SO1[1].T, W2),
                                     np.linalg.inv(t2))

            # U3
            W3 = sl.khatri_rao(patch.factor.U4, patch.factor.U2)
            W3 = sl.khatri_rao(W3, patch.factor.U1)
            leftD = np.linalg.inv(2 * lda * np.dot(R.T, R) + mu * np.eye(R.shape[1]))
            rightD = np.linalg.inv(np.dot(W3.T, W3))
            t3 = np.dot(2 * lda * np.dot(R.T, YSO1[2].T) + mu * SO1[2].T, W3)
            patch.factor.U3 = np.dot(leftD, np.dot(t3, rightD))

            # U4
            W4 = sl.khatri_rao(patch.factor.U3, patch.factor.U2)
            W4 = sl.khatri_rao(W4, patch.factor.U1)
            P4 = sl.khatri_rao(np.dot(R, patch.factor.U3), patch.factor.U2)
            P4 = sl.khatri_rao(P4, patch.factor.U1)
            G4 = np.dot(patch.factor.U3.T, patch.factor.U3) * np.dot(patch.factor.U2.T, patch.factor.U2) \
                 * np.dot(patch.factor.U1.T, patch.factor.U1)
            P4TP4 = np.dot(np.dot(R, patch.factor.U3).T, np.dot(R, patch.factor.U3)) * np.dot(patch.factor.U2.T,
                                                                                              patch.factor.U2) \
                    * np.dot(patch.factor.U1.T, patch.factor.U1)
            t4 = mu * G4 + 2 * lda * P4TP4
            patch.factor.U4 = np.dot(np.dot(MM2[3].T, W4) + 2 * lda * np.dot(YSO1[3].T, P4) + mu * np.dot(SO1[3].T, W4),
                                     np.linalg.inv(t4))
            factor = patch.factor
            cube = ktensor([factor.U1, factor.U2, np.dot(R, patch.factor.U3), factor.U4])
            err = lda * np.linalg.norm(cube - Ytt1)
            print(patch.Rank, err)

            # U1 = patch.factor.U1
            # U2 = patch.factor.U2
            # U3 = patch.factor.U3
            # U4 = patch.factor.U4
            #
            # for i in range(patch.Rank + 1, 100):
            #     # rank 更新
            #
            #     new_column = np.random.random([U1.shape[0], 1])
            #     U1 = np.concatenate((U1, new_column), axis=1)
            #
            #     new_column = np.random.random([U2.shape[0], 1])
            #     U2 = np.concatenate((U2, new_column), axis=1)
            #
            #     new_column = np.random.random([U3.shape[0], 1])
            #     U3 = np.concatenate((U3, new_column), axis=1)
            #
            #     new_column = np.random.random([U4.shape[0], 1])
            #     U4 = np.concatenate((U4, new_column), axis=1)
            #     #
            #     # U1
            #     W1 = sl.khatri_rao(U4, U3)
            #     W1 = sl.khatri_rao(W1, U2)
            #     P1 = sl.khatri_rao(U4, np.dot(R, U3))
            #     P1 = sl.khatri_rao(P1, U2)
            #
            #     G1 = np.dot(U4.T, U4) * np.dot(U3.T, U3) * np.dot(U2.T, U2)
            #     PTP = np.dot(U4.T, U4) * np.dot(np.dot(R, U3).T, np.dot(R, U3)) * np.dot(U2.T, U2)
            #     t1 = mu * G1 + 2 * lda * PTP
            #     # np.dot(patch.M2[0].T, W1)
            #     U1 = np.dot(np.dot(MM2[0].T, W1) + 2 * lda * np.dot(YSO1[0].T, P1) +
            #                 mu * np.dot(SO1[0].T, W1), np.linalg.inv(t1))
            #
            #     # U2
            #     W2 = sl.khatri_rao(U4, U3)
            #     W2 = sl.khatri_rao(W2, U1)
            #     P2 = sl.khatri_rao(U4, np.dot(R, U3))
            #     P2 = sl.khatri_rao(P2, U1)
            #     G2 = np.dot(U4.T, U4) * np.dot(U3.T, U3) * np.dot(U1.T, U1)
            #     P2TP2 = np.dot(U4.T, U4) * np.dot(np.dot(R, U3).T, np.dot(R, U3)) * np.dot(U1.T, U1)
            #     t2 = mu * G2 + 2 * lda * P2TP2
            #     U2 = np.dot(np.dot(MM2[1].T, W2) + 2 * lda * np.dot(YSO1[1].T, P2) + mu * np.dot(SO1[1].T, W2),
            #                 np.linalg.inv(t2))
            #
            #     # U3
            #     W3 = sl.khatri_rao(U4, U2)
            #     W3 = sl.khatri_rao(W3, U1)
            #     leftD = np.linalg.inv(2 * lda * np.dot(R.T, R) + mu * np.eye(R.shape[1]))
            #     rightD = np.linalg.inv(np.dot(W3.T, W3))
            #     t3 = np.dot(2 * lda * np.dot(R.T, YSO1[2].T) + mu * SO1[2].T, W3)
            #     U3 = np.dot(leftD, np.dot(t3, rightD))
            #
            #     # U4
            #     W4 = sl.khatri_rao(U3, U2)
            #     W4 = sl.khatri_rao(W4, U1)
            #     P4 = sl.khatri_rao(np.dot(R, U3), U2)
            #     P4 = sl.khatri_rao(P4, U1)
            #     G4 = np.dot(U3.T, U3) * np.dot(U2.T, U2) * np.dot(U1.T, U1)
            #     P4TP4 = np.dot(np.dot(R, U3).T, np.dot(R, U3)) * np.dot(U2.T, U2) * np.dot(U1.T, U1)
            #     t4 = mu * G4 + 2 * lda * P4TP4
            #     U4 = np.dot(np.dot(MM2[3].T, W4) + 2 * lda * np.dot(YSO1[3].T, P4) + mu * np.dot(SO1[3].T, W4),
            #                 np.linalg.inv(t4))
            #
            #     cube = ktensor([U1, U2, np.dot(R, U3), U4])
            #     err1 = lda * np.linalg.norm(cube - Ytt1)
            #
            #     if (np.abs(err - err1) < 1):
            #         print(patch.Rank, err)
            #         break
            #     err = err1
            #     patch.Rank = patch.Rank + 1
            # patch.factor.U1 = U1
            # patch.factor.U2 = U2
            # patch.factor.U3 = U3
            # patch.factor.U4 = U4
            factor = patch.factor
            patches = ktensor([factor.U2, factor.U1, factor.U3, factor.U4])
            for ind_cur, index in enumerate(indices):
                row = int(index % rows)
                col = int((index - row) / rows)
                # if(row == 0 and col == 0):
                #     print(patches[:, :, 1, ind_cur])
                E_Img[row: patsize + row, col: patsize + col, :] = \
                    E_Img[row: patsize + row, col: patsize + col, :] + patches[:, :, :, ind_cur]

            curPatchlist.append(patch)
            # print("----end---")
        # print(len(curPatchlist))
        time_end = time.time()
        print("upate 分区更新完成，用时%f秒" % (time_end - time_start))

        # sparse

        sparseTensor = SparseTensor()
        #
        data = []
        # csr_first = csr_matrix(E_Img[:, :, 0])
        #
        for i in range(nn[2]):
            print(E_Img[:, :, i].shape)
            csr = csr_matrix(E_Img[:, :, i])
            sparseTensor.addIndices(csr.indices)
            sparseTensor.addOffset(csr.indptr)
            data.append(csr.data)
        sparseTensor.addData(data)
        #

        print("零占比：", 1 - np.count_nonzero(E_Img[:, :, 0]) / nn[0] / nn[1])

        # self.E_Img = E_Img
        # return E_Img
        self.E_Img = sparseTensor

    def reduce(self, data):
        self.E_Img = self.E_Img + data


    # def updateEimg(self, *parameterServer):
    #     parameterServer.put_HR_HSI.remote(self.E_Img)


    def getEimg(self):
        return self.E_Img