import numpy as np
import ray

@ray.remote(num_cpus=1)
class TestActor:
    def __init__(self, nn):
        self.E = np.ones(nn)
        print("初始化")
    def update(self):
        print("actor更新")
        return self.E

