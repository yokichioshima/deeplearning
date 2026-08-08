import sys
sys.path.append('..')
import numpy as np
from layers.softmax import Softmax

class AttentionWeight:
    # 初期化。
    # param: self: AttentionWeight layer。
    def __init__(self):
        self.params, self.grads = [], []
        self.softmax = Softmax()
        self.cache = None
    
    # 順伝播。
    # param: self: AttentionWeight layer。
    # param: hs: Encoder からの隠れ状態の配列。(batch_size, time_steps, hidden_size) の形状を持つ。
    # param: h: 下流からの隠れ状態。(batch_size, hidden_size) の形状を持つ。
    # return: a: 各単語の重み。(batch_size, time_steps) の形状を持つ。
    def forward(
        self, 
        hs: np.ndarray[np.float32], 
        h: np.ndarray[np.float32]    
    ) -> np.ndarray[np.float32]:
        N, T, H = hs.shape

        hr = h.reshape(N, 1, H)

        # ブロードキャストのより、hr は (N, T, H) の形状に拡張
        t = hs * hr
        s = np.sum(t, axis=2)
        a = self.softmax.forward(s)

        self.cache = (hs, hr)
        return a
    
    # 逆伝播。
    # param: self: AttentionWeight layer。
    # param: da: 各単語の重みの勾配。(batch_size, time_steps) の形状を持つ。
    # return: dhs: Encoder からの隠れ状態の配列の勾配。(batch_size, time_steps, hidden_size) の形状を持つ。
    # return: dh: 下流からの隠れ状態の勾配。(batch_size, time_steps) の形状を持つ。
    def backward(
        self,
        da: np.ndarray[np.float32]
    ) -> tuple[
        np.ndarray[np.float32],
        np.ndarray[np.float32]
    ]:
        hs, hr = self.cache
        N, T, H = hs.shape

        ds = self.softmax.backward(da)
        dt = ds.reshape(N, T, 1).repeat(H, axis=2)
        dhs = dt * hr
        dhr = dt * hs
        dh = np.sum(dhr, axis=1)

        return dhs, dh