import time

import ray
import numpy as np


@ray.remote
class ParameterServerActor(object):
    def __init__(self, nn):
        self.HR_HSI = np.zeros(nn)

    def put_HR_HSI(self, *HSI_Piece):
        # print(HSI_Piece)
        self.HR_HSI = np.zeros(self.HR_HSI.shape)
        for hsi in HSI_Piece:
            print("我在更新", time.ctime())
            self.HR_HSI += hsi

        print("尺寸", self.HR_HSI.shape)
        return self.HR_HSI
        # self.net.apply_gradients(np.mean(gradients, axis=0))
        # return self.net.variables.get_flat()

    def get_HR_HSI(self):
        return self.HR_HSI

    def reset(self):
        self.HR_HSI = np.zeros(self.HR_HSI.shape)


