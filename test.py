import pickle
import time
import numpy as np
import ray
import sys
import os

sys.path.append('./utils')
sys.path.append('./classes')
sys.path.append('./actors')
sys.path.append('./tasks')
#

from actors.TestActor import TestActor

from actors.ParameterServerActor import ParameterServerActor
from utils.data import readData, loads
from ANPTCP4 import test
from utils.measurement_function import QualityIndices, PSNR3D

# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    # 初始化Ray
    ray.init(address="ray://10.37.129.13:10001",
             runtime_env={"working_dir": "./"})
    nn = [1860, 680, 186]



    # 创建MatrixSumActor的实例，每个Actor存储一个矩阵
    factorActors = [TestActor.remote(nn) for i in range(5)]

    pieceHSI = [actor.update.remote() for actor in factorActors]

    ps = ParameterServerActor.remote(nn)

    print("我结束了", time.ctime())

    HT = ray.get(ps.put_HR_HSI.remote(*pieceHSI)).copy()
    print(HT)
