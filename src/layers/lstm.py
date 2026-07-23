import numpy as np
from common.functions import sigmoid

class LSTM:
    # LSTM layer の初期化。
    # param: self: LSTM layer。
    # param: Wx: 入力 x に対する重み。(input_size, 4 * hidden_size) の形状を持つ。
    # param: Wh: 前の時刻の隠れ層に対する重み。(hidden_size, hidden_size) の形状を持つ。
    # param: b: バイアス。(4 * hidden_size,) の形状を持つ。 
    def __init__(
        self, 
        Wx: np.ndarray[np.float32], 
        Wh: np.ndarray[np.float32], 
        b: np.ndarray[np.float32]
    ):
        self.params = [Wx, Wh, b]
        self.grads = [np.zeros_like(Wx), np.zeros_like(Wh), np.zeros_like(b)]
        self.cache = None
    
    # 順伝播。
    # param: self: LSTM layer。
    # param: x: 入力データ。(batch_size, input_size) の形状を持つ。
    # param: h_prev: 前の時刻の隠れ状態。(batch_size, hidden_size) の形状を持つ。
    # param: c_prev: 前の時刻の記憶セル。(batch_size, hidden_size) の形状を持つ。
    # return h_next: 隠れ状態。(batch_size, hidden_size) の形状を持つ。
    # return c_next: 記憶セル。(batch_size, hidden_size) の形状を持つ。
    def forward(
        self,
        x, 
        h_prev, 
        c_prev
    ) -> tuple[
        np.ndarray[np.float32],
        np.ndarray[np.float32]
    ]:
        Wx, Wh, b = self.params
        N, H = h_prev.shape

        A = np.dot(x, Wx) + np.dot(h_prev, Wh) + b

        #slice
        f = A[:, :H]
        g = A[:, H:2*H]
        i = A[:, 2*H:3*H]
        o = A[:, 3*H:]

        f = sigmoid(f)
        g = np.tanh(g)
        i = sigmoid(i)
        o = sigmoid(o)

        c_next = f * c_prev + g * i
        h_next = o * np.tanh(c_next)

        self.cache = (x, h_prev, c_prev, i, f, g, o, c_next)
        return h_next, c_next
    
    # 逆伝播。
    # param: self: LSTM layer。
    # param: dh_next: 隠れ状態の勾配。(batch_size, hidden_size) の形状を持つ。
    # param: dc_next: 記憶セルの勾配。(batch_size, hidden_size) の形状を持つ。
    # return: dx: 入力データの勾配。(batch_size, input_size) の形状を持つ。
    # return: dh_prev: 前の時刻の隠れ状態の勾配。(batch_size, hidden_size) の形状を持つ。
    # return: dc_prev: 前の時刻の記憶セルの勾配。(batch_size, hidden_size) の形状を持つ。
    def backward(
        self, 
        dh_next: np.ndarray[np.float32], 
        dc_next: np.ndarray[np.float32]
    ) -> tuple[
        np.ndarray[np.float32],
        np.ndarray[np.float32],
        np.ndarray[np.float32]
    ]:
        Wx, Wh, b = self.params
        x, h_prev, c_prev, i, f, g, o, c_next = self.cache

        tanh_c_next = np.tanh(c_next)        
        ds = dc_next + dh_next * o * (1 - tanh_c_next ** 2)

        dc_prev = f * ds
        
        di = g * ds
        df = c_prev * ds
        do = dh_next * tanh_c_next
        dg = i * ds

        di *= i * (1 - i)
        df *= f * (1 - f)
        do *= o * (1 - o)
        dg *= (1 - g ** 2)

        dA = np.hstack((df, dg, di, do))

        dWh = np.dot(h_prev.T, dA)
        dWx = np.dot(x.T, dA)
        db = dA.sum(axis=0)

        self.grads[0][...] = dWx
        self.grads[1][...] = dWh
        self.grads[2][...] = db

        dx = np.dot(dA, Wx.T)
        dh_prev = np.dot(dA, Wh.T)

        return dx, dh_prev, dc_prev
