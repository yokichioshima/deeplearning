import sys
sys.path.append('..')
import numpy as np
from layers.attention_weight import AttentionWeight
from layers.weight_sum import WeightSum

class Attention:
    # 初期化。
    # param: self: Attention layer。
    def __init__(self):
        self.params, self.grads = [], []
        self.attention_weight_layer = AttentionWeight()
        self.weight_sum_layer = WeightSum()
        self.attention_weight = None
    
    # 順伝播。
    # param: self: Attention layer。
    # param: hs: Encoder からの隠れ状態の配列。(batch_size, time_steps, hidden_size) の形状を持つ。
    # param: h: 下流からの隠れ状態。(batch_size, hidden_size) の形状を持つ。
    # return: out: コンテキストベクトル。(batch_size, hidden_size) の形状を持つ。
    def forward(
        self,
        hs: np.ndarray[np.float32],
        h: np.ndarray[np.float32]
    ) -> np.ndarray[np.float32]:
        a = self.attention_weight_layer.forward(hs, h)
        out = self.weight_sum_layer.forward(hs, a)
        self.attention_weight = a
        return out
    
    # 逆伝播。
    # param: self: Attention layer。
    # param: dout: コンテキストベクトルの勾配。
    # return: dhs: Encoder からの隠れ状態の配列の勾配。(batch_size, time_steps, hidden_size) の形状を持つ。
    # return: dh: 下流からの隠れ状態の勾配。(batch_size, hidden_size) の形状を持つ。
    def backward(
        self,
        dout: np.ndarray[np.float32]
    ) -> tuple[
        np.ndarray[np.float32],
        np.ndarray[np.float32]
    ]:
        dhs0, da = self.weight_sum_layer.backward(dout)
        dhs1, dh = self.attention_weight_layer.backward(da)
        dhs = dhs0 + dhs1
        return dhs, dh
