import numpy as np
import scipy.io as sio
import h5py
import platform
from utils.tools import gaussian, downsample

dir = "/data2/data/"
system = platform.system()
if system == "Windows":
    dir = "D:/博士生涯/云计算/我的论文/tgrs/DPN4CTCP/data/"
# read data from /data
def readData(filename):
    R = loadR()
    s = loads()

    if(filename == "DC50"):
        hf = h5py.File(dir + "dc50.mat", 'r')
        HH = np.array(hf["I_REF"]).T
        nn = HH.shape
    elif filename == 'DC':
        hf = h5py.File(dir + "I_REF.mat", 'r')
        HH = np.array(hf["I_REF"]).T
        nn = HH.shape
    elif (filename == "P"):
        hf = h5py.File(dir + "Pavia_HH.mat", 'r')
        HH = np.array(hf["I_REF"]).T
        nn = HH.shape
        R = R[:, 0: nn[2]]
    elif (filename == "M"):
        hf = h5py.File(dir + "m.mat", 'r')
        HH = np.array(hf["I_REF"]).T
        HH = HH[0: 1860, 0: 680, :]
        hf = h5py.File(dir + "R_186.mat", 'r')
        R = np.array(hf["R"]).T
        nn = HH.shape
        R = R[:, 0: nn[2]]
    else:
        print("filename must be DC50, DC, and P")
        return
    I_temp = np.reshape(HH, [nn[0] * nn[1], nn[2]], order='F')
    I_ms = np.dot(R, I_temp.T)
    MSI = np.reshape(I_ms.T, [nn[0], nn[1], R.shape[0]], order='F')

    I_HSn = gaussian(HH, s)
    HSI = downsample(I_HSn, 5)

    return HH, MSI, HSI, R


def loadR():
    hf = h5py.File(dir + "R.mat", 'r')
    R = np.array(hf["R"])
    return R.T


def loads():
    hf = h5py.File(dir + "s.mat", 'r')
    R = np.array(hf["s"])
    return R.T