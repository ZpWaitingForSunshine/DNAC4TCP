import numpy as np
from scipy.sparse import csr_matrix

# 创建一个密集矩阵
dense_matrix = np.array([[0, 3, 0], [0, 0, 0], [0, 0, 0]])

# 将密集矩阵转换为CSR格式的稀疏矩阵
sparse_matrix = csr_matrix(dense_matrix)

print("原始密集矩阵:")
print(dense_matrix)

print("\n转换后的稀疏矩阵 (CSR格式):")
print(sparse_matrix)


c = csr_matrix((sparse_matrix.data, sparse_matrix.indices,
                                      sparse_matrix.indptr)).toarray()
print(c)