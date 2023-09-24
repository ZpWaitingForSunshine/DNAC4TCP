import logging

import numpy as np

from utils.tensor_function import ktensor


def CC(ref, tar):
    rows, cols, bands = tar.shape
    out = np.zeros(bands)

    for i in range(bands):
        tar_tmp = tar[:, :, i]
        ref_tmp = ref[:, :, i]
        cc = np.corrcoef(tar_tmp.ravel(), ref_tmp.ravel())
        out[i] = cc[0, 1]

    return np.mean(out)

def SAM(ref, tar):
    rows, cols, bands = tar.shape
    prod_scal = np.sum(ref * tar, axis=2)
    norm_orig = np.sum(ref * ref, axis=2)
    norm_fusa = np.sum(tar * tar, axis=2)
    prod_norm = np.sqrt(norm_orig * norm_fusa)
    prod_map = prod_norm.copy()
    prod_map[prod_map == 0] = np.finfo(float).eps
    # map = np.arccos(prod_scal / prod_map)
    prod_scal = prod_scal.ravel()
    prod_norm = prod_norm.ravel()
    z = np.where(prod_norm == 0)
    prod_scal = np.delete(prod_scal, z)
    prod_norm = np.delete(prod_norm, z)
    angle_SAM = np.sum(np.arccos(prod_scal / prod_norm)) * (180 / np.pi) / len(prod_norm)

    return angle_SAM

def RMSE(ref, tar):
    rows, cols, bands = ref.shape
    out = np.sqrt(np.sum(np.sum(np.sum((tar - ref) ** 2))) / (rows * cols * bands))
    return out


def ERGAS(I, I_Fus, Resize_fact):
    I = I.astype(np.float64)
    I_Fus = I_Fus.astype(np.float64)

    Err = I - I_Fus
    ERGAS = 0
    for iLR in range(Err.shape[2]):
        ERGAS += np.mean(Err[:, :, iLR] ** 2) / np.mean(I[:, :, iLR]) ** 2

    ERGAS = (100 / Resize_fact) * np.sqrt((1 / Err.shape[2]) * ERGAS)

    return ERGAS


def PSNR3D(imagery1, imagery2):
    m, n, k = imagery1.shape
    mm, nn, kk = imagery2.shape
    m = min(m, mm)
    n = min(n, nn)
    k = min(k, kk)
    imagery1 = imagery1[0:m, 0:n, 0:k]
    imagery2 = imagery2[0:m, 0:n, 0:k]
    psnr = 0
    for i in range(k):
        mse = np.mean((imagery1[:, :, i] - imagery2[:, :, i]) ** 2)
        psnr += 10 * np.log10(255 ** 2 / mse)
    psnr /= k

    return psnr


def loss(U1, U2, U3, U4, Y, lda, X, mu, M):
    L1 = lda * np.linalg.norm(ktensor([U1, U2, U3, U4]) - Y)
    # L2 = mu / 2 * np.linalg.norm(ktensor([U1, U2, U3]) - X - M/mu)
    return L1

def QualityIndices(I_HS, I_REF, ratio):
    rows, cols, bands = I_REF.shape
    I_HS = I_HS[ratio: rows - ratio, ratio: cols - ratio,:]
    I_REF = I_REF[ratio :rows - ratio, ratio: cols - ratio,:]
    cc = CC(I_HS, I_REF)
    logging.info("cc: ", cc)
    print("cc: ", cc)
    sam = SAM(I_HS, I_REF)
    print('sam: ', sam)
    logging.info('sam: ', sam)


    rmse = RMSE(I_HS, I_REF)
    print('rmse: ', rmse)
    logging.info('rmse: ', rmse)

    ergas = ERGAS(I_HS, I_REF, ratio)
    print('ERGAS: ', ergas)
    logging.info('ERGAS: ', ergas)