import sys
sys.path.append('..')
import numpy as np
from layers.attention import Attention

class TimeAttention:
    # 初期化。
    # param: self: TimeAttention layer。
    def __init__(self):
        self.params, self.grads = [], []
        self.layers = None
        self.attention_weights = None
    
    # 順伝播。
    # param: self: TimeAttention layer。
    # param: hs_enc: Encoder からの隠れ状態の配列。(batch_size, time_steps, hidden_size) の形状を持つ。
    # param: hs_dec: 下流からの隠れ状態。(batch_size, time_steps, hidden_size) の形状を持つ。
    # return: out: コンテキストベクトルの配列。(batch_size, time_steps, hidden_size) の形状を持つ。
    def forward(
        self, 
        hs_enc: np.ndarray[np.float32], 
        hs_dec: np.ndarray[np.float32]
    ) -> np.ndarray[np.float32]:
        N, T, H = hs_dec.shape
        out = np.empty_like(hs_dec)
        self.layers = []
        self.attention_weights = []

        for t in range(T):
            layer = Attention()
            out[:, t, :] = layer.forward(hs_enc, hs_dec[:, t, :])
            self.layers.append(layer)
            self.attention_weights.append(layer.attention_weight)
        
        return out
    
    # 逆伝播。
    # param: self: TimeAttention layer。
    # param: dout: コンテキストベクトルの配列の勾配。(batch_size, time_steps, hidden_size) の形状を持つ。
    # return: dhs_enc: Encoder からの隠れ状態の配列の勾配。(batch_size, time_steps, hidden_size) の形状を持つ。
    # return: dhs_dec: 下流からの隠れ状態の配列の勾配。 (batch_size, time_steps, hidden_size) の形状を持つ。
    def backward(
        self, 
        dout: np.ndarray[np.float32]
    ) -> np.ndarray[np.float32]:
        N, T, H = dout.shape
        dhs_enc = 0
        dhs_dec = np.empty_like(dout)

        for t in range(T):
            layer = self.layers[t]
            dhs, dh = layer.backward(dout[:, t, :])
            dhs_enc += dhs
            dhs_dec[:, t, :] = dh
        
        return dhs_enc, dhs_dec