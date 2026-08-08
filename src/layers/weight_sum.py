import sys
sys.path.append('..')
import numpy as np
from layers.softmax import Softmax

class WeightSum:
    # 初期化。
    # param: self: weight_sum layer。
    def __init__(self):
        self.params, self.grads = [], []
        self.cache = None
    
    # 順伝播。
    # param: self: weight_sum layer。
    # param: hs: 隠れ状態の配列。(batch_size, time_steps, hidden_size) の形状を持つ。
    # param: a: 重みの配列。(batch_size, time_steps) の形状を持つ。
    def forward(
        self, 
        hs: np.ndarray[np.float32], 
        a: np.ndarray[np.float32]
    ) -> np.ndarray[np.float32]:
        N, T, H = hs.shape

        ar = a.reshape(N, T, 1)

        # ブロードキャストにより、ar の形状が (N, T, H) に拡張される。
        t = hs * ar 
        c = np.sum(t, axis=1)

        self.cache = (hs, ar)
        return c
    
    # 逆伝播。
    # param: self: weight_sum layer。
    # param: dc: 上流からの勾配。(batch_size, hidden_size) の形状を持つ。
    # return: dhs: 隠れ状態の勾配。(batch_size, time_steps, hidden_size) の形状を持つ。
    # return: da: 重みの勾配。(batch_size, time_steps) の形状を持つ。
    def backward(
        self, 
        dc: np.ndarray[np.float32]
    ) -> tuple[np.ndarray[np.float32], np.ndarray[np.float32]]:
        hs, ar = self.cache
        N, T, H = hs.shape
        dt = dc.reshape(N, 1, H).repeat(T, axis=1)
        dhs = dt * ar
        dar = dt * hs
        da = np.sum(dar, axis=2)
        
        return dhs, da