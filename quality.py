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
    I_REF, MSI, HSI, R = readData('M')






