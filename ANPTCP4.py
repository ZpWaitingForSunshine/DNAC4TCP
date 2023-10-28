import os.path
import sys
import time

import pickle
import numpy as np
import ray
import matplotlib.pyplot as plt

from actors.FactorActor import FactorActor
from utils.nonlocal_function import find_min_indices, split_average, getW_Imge_Matrix
from utils.tools import upsample, gaussian, average_split_set
from tasks.GroupTask import knn, partitions_group, cg
from actors.ParameterServerActor import ParameterServerActor

def group(PN, num, rows, cols, patsize, Y_ref):
    indices = np.zeros((2, rows * cols)).astype('int')
    indices[0] = np.arange(rows * cols)
    indices[1] = 1000000

    groups_edges = split_average(indices[0, :], PN, num)
    indices_set = []
    # Y = ray.get(Y_ref)
    indices_set = []
    for i in range(int(np.ceil(rows * cols / PN))):
        indices_split_arrays = np.array_split(indices, num, axis=1)
        task_ids = [knn.remote(item, rows, cols, patsize, indices[0][0], Y_ref) for item in indices_split_arrays]
        indices_list = ray.get(task_ids)
        indices = np.concatenate(indices_list, axis=1)
        min_indices = find_min_indices(indices, PN)
        indices_set.append(indices[0, min_indices])
        indices = np.delete(indices, min_indices, axis=1)
    indices_set.append(indices[0, :])

    groups = average_split_set(indices_set, num)
    # 创建任务
    # task_ids = [partitions_group.remote(item, rows, cols, patsize, Y_ref, PN) for item in indices_set]
    # groups = ray.get(task_ids)
    return groups

def test(Ob, KK, Y, rate, PN, R, s, maxIter, num):
    print("start")

    time_nonlocal = 0
    time_updateX = 0
    time_group = 0
    time_reduce = 0




    t_start = time.time()
    print("")
    max_HS = np.max(Ob)
    Ob = Ob / max_HS
    Y = Y / max_HS

    Z_ref = ray.put(Ob)
    Y_ref = ray.put(Y)

    Ob = upsample(Ob, rate)
    # % parameter
    tol = 1e-2
    mu = 1e-4
    lda = 100
    maxIter = 1
    minIter = 1
    patsize = 5
    Pstep = 1

    Z = Ob
    M1 = np.zeros(Z.shape)
    Ob = gaussian(Ob, s)

    M1_ref = ray.put(M1)

    nn = Ob.shape
    rows = nn[0] - patsize + 1
    cols = nn[1] - patsize + 1


    # 初始化一个参数服务器
    parameterServerActor = ParameterServerActor.remote(nn)

    t1 = time.time()



    W_Img = getW_Imge_Matrix(nn, rows, cols, patsize)

    X_ref = ray.put(Ob)

    # if(os.path.exists("./k1")):
    #     with open('./k1', 'rb') as f:
    #         k1 = pickle.load(f)
    # else:
    k1 = group(PN, num, rows, cols, patsize, Y_ref)
    #     with open('./k1', 'wb') as file:
    #         pickle.dump(k1, file)

            # K1 = ['sfsdfdf']
            #
            # if (os.path.exists("./k")):
            #     with open('./k', 'rb') as f:
            #         k1 = pickle.load(f)
            # else:
            #     # k1 = group(PN, num, rows, cols, patsize, Y_ref)
            #     with open('./k', 'wb') as file:
            #         pickle.dump(K1, file)


    t2 = time.time()

    time_nonlocal = t2 - t1

    print("nonlocal数据：", time_nonlocal, "s")

    # 创建MatrixSumActor的实例，每个Actor存储一个矩阵
    factorActors = [FactorActor.remote(item, Y_ref, patsize, rows, cols, nn) for item in k1]


    for i in range(maxIter):

        t3 = time.time()
        # update factors
        ray.wait([actor.updateFactors.remote(X_ref, patsize,
                          rows, cols, M1_ref, Y_ref, lda, mu, R, nn) for actor in factorActors])
        # # fold_actors = [actor.reduce.remote() for actor in factorActors]

        t4 = time.time()
        time_group = time_group + t4 - t3

        pieceHSI = [actor.getEimg.remote() for actor in factorActors]

        t31 = time.time()
        # ray.get([actor.updateEimg.remote(parameterServerActor) for actor in factorActors])
        HT = ray.get(parameterServerActor.put_HR_HSI.remote(*pieceHSI)).copy()
        print("HT更新结束")
        # HT = ray.get(parameterServerActor.get_HR_HSI)

        # fold_actor = factorActors[0]
        # for ii, other_actor in enumerate(factorActors):
        #     if ii != 0:
        #         ray.get(fold_actor.reduce.remote(other_actor.getEimg.remote()))
        t32 = time.time()

        time_reduce = time_reduce + t32 - t31
        # HT = ray.get(factorActors[0].getEimg.remote()).copy()
        # HT = ray.get(parameterServerActor.get_HR_HSI.remote())
        for band in range(nn[2]):
            HT[:, :, band] = HT[:, :, band] / (W_Img + np.finfo(float).eps)
        # plt.imshow(HT[:, :, 1:4])
        # plt.show()

        rr = 2 * Ob + mu * HT - M1
        rr_ref = ray.put(rr)

        t5 = time.time()
        list_indices = split_average(range(nn[2]), 1, num)
        x_tasks = [cg.remote(item, rr_ref, mu, rate, s) for item in list_indices]
        x_tasks_set = ray.get(x_tasks)
        Z = np.concatenate(x_tasks_set, axis=2)

        t6 = time.time()

        time_updateX = time_updateX + t6 - t5

        print("Z更新结束")

        # plt.imshow(Z[:, :, 1:4])
        # plt.show()
        M1 = M1 + mu * (Z - HT)


        M1_ref = ray.put(M1)
        X_ref = ray.put(Z)


    print("nonlocal数据：", time_nonlocal, "s")
    print("time_updateX：", time_updateX, "s")
    print("time_group：", time_group, "s")
    print("time_reduce：", time_reduce, "s")

    return Z * max_HS










