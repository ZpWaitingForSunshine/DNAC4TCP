import pickle
import time
import numpy as np
import ray
import sys
sys.path.append('./utils')
sys.path.append('./classes')
sys.path.append('./actors')
sys.path.append('./tasks')
#



from utils.data import readData, loads
from ANPTCP4 import test
from utils.measurement_function import QualityIndices, PSNR3D

# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    # 初始化Ray
    # ray.init(address="ray://10.37.129.13:10001",
    #          runtime_env={"working_dir": "./"})
    ray.init()

    # read data 读取数据
    I_REF, MSI, HSI, R = readData('DC50')

    PN = 300
    Rank = 1
    ratio = 5
    s = loads()
    t1 = time.time()

    KK = 80

    rate = 5
    maxIter = 2
    num = 5

    I_CTD = test(HSI, KK, MSI, rate, PN, R, s, maxIter, num)

    QualityIndices(I_CTD, I_REF, ratio)
    AM = np.max(I_REF)
    psnr = PSNR3D(I_CTD * 255 / AM, I_REF * 255 / AM)
    print('psnr: ', psnr)

    t2 = time.time()
    print('time(s): ', t2 - t1)
    filename = "x_3order.pkl"

    with open(filename, 'wb') as file:
        pickle.dump(I_CTD, file)


