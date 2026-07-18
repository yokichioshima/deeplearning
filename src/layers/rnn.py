import numpy as np

class RNN:
    # RNN layer の初期化。
    # param: Wx: 入力 x に対する重み。(input_size, hidden_size) の形状を持つ。
    # param: Wh: 前の時刻の隠れ状態 h に対する重み。(hidden_size, hidden_size) の形状を持つ。
    # param: b: バイアス。(hidden_size,) の形状を持つ。
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
    # param: self: RNN layer。
    # param: x: 入力データ。(batch_size, input_size) の形状を持つ。
    # param: h_prev: 前の時刻の隠れ状態。(batch_size, hidden_size) の形状を持つ。
    # return: h_next: 次の時刻の隠れ状態。(batch_size, hidden_size) の形状を持つ。
    def forward(
        self,
        x: np.ndarray[np.float32],
        h_prev: np.ndarray[np.float32]
    ) -> np.ndarray[np.float32]:
        Wx, Wh, b = self.params
        t = np.dot(h_prev, Wh) + np.dot(x, Wx) + b
        h_next = np.tanh(t)

        self.cache = (x, h_prev, h_next)
        return h_next
    
    # 逆伝播。
    # param: self: RNN layer。
    # param: dh_next: 次の時刻の隠れ状態の勾配。(batch_size, hidden_size) の形状を持つ。
    # return: dx: 入力データの勾配。(batch_size, input_size) の形状を持つ。
    # return: dh_prev: 前の時刻の隠れ状態の勾配。(batch_size, hidden_size) の形状を持つ。
    def backward(
        self,
        dh_next: np.ndarray[np.float32]
    ) -> tuple[
        np.ndarray[np.float32],
        np.ndarray[np.float32]
        ]:
        Wx, Wh, b = self.params
        x, h_prev, h_next = self.cache

        dt = dh_next * (1 - h_next ** 2)
        db = np.sum(dt, axis=0)
        dWh = np.dot(h_prev.T, dt)
        dh_prev = np.dot(dt, Wh.T)
        dWx = np.dot(x.T, dt)
        dx = np.dot(dt, Wx.T)

        self.grads[0][...] = dWx
        self.grads[1][...] = dWh
        self.grads[2][...] = db

        return dx, dh_prev