import numpy as np
import tensorly as tl




def tensor_mode_unfolding(tensor, mode):
    """Performs mode unfolding of a tensor along a given mode in row-major order."""
    shape = tensor.shape
    mode_unfolded = np.moveaxis(tensor, mode, 0)
    mode_unfolded = np.reshape(mode_unfolded, (shape[mode], -1), order='F')
    return mode_unfolded


def matricize(tensor):
    dim = np.ndim(tensor)
    data = []
    for i in range(dim):
        data.append(tensor_mode_unfolding(tensor, mode=i).T)
    return data


def ktensor(Us):
    R = Us[0].shape[1]
    cp_tensor = (np.ones(R), Us)
    return tl.cp_to_tensor(cp_tensor)