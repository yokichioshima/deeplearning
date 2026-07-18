import numpy as np
from layers.rnn import RNN

class TimeRNN:
    # TimeRNN layer の初期化。
    # param: Wx: 入力 x に対する重み。(input_size, hidden_size) の形状を持つ。
    # param: Wh: 前の時刻の隠れ状態 h に対する重み。(hidden_size, hidden_size) の形状を持つ。
    # param: b: バイアス。(hidden_size,) の形状を持つ。
    # param: stateful: True の場合、隠れ状態を保持する。
    def __init__(
        self,
        Wx: np.ndarray[np.float32],
        Wh: np.ndarray[np.float32],
        b: np.ndarray[np.float32],
        stateful: bool = False
    ):
        self.params = [Wx, Wh, b]
        self.grads = [np.zeros_like(Wx), np.zeros_like(Wh), np.zeros_like(b)]
        self.layers = None
        self.h, self.dh = None, None
        self.stateful = stateful
    
    def set_state(self, h: np.ndarray[np.float32]):
        self.h = h
    
    def reset_state(self):
        self.h = None
    
    # 順伝播。
    # param: self: TimeRNN layer。
    # param: xs: 入力データ。(batch_size, time_steps, input_size) の形状を持つ。
    # return: hs: 隠れ状態。(batch_size, time_steps, hidden_size) の形状を持つ。
    def forward(
        self,
        xs: np.ndarray[np.float32]
    ) -> np.ndarray[np.float32]:
        Wx, Wh, b = self.params
        N, T, D = xs.shape
        H = Wh.shape[0]

        self.layers = []
        hs = np.empty((N, T, H), dtype='f')

        if not self.stateful or self.h is None:
            self.h = np.zeros((N, H), dtype='f')

        for t in range(T):
            layer = RNN(*self.params)
            self.h = layer.forward(xs[:, t, :], self.h)
            hs[:, t, :] = self.h
            self.layers.append(layer)

        return hs
    
    # 逆伝播。
    # param: self: TimeRNN layer。
    # param: dhs: 隠れ状態の勾配。(batch_size, time_steps, hidden_size) の形状を持つ。
    # return: dxs: 入力データの勾配。(batch_size, time_steps, input_size) の形状を持つ。
    def backward(
        self,
        dhs: np.ndarray[np.float32]
    ) -> np.ndarray[np.float32]:
        Wx, Wh, b = self.params
        N, T, H = dhs.shape
        D, H = Wx.shape

        dxs = np.empty((N, T, D), dtype='f')
        dh = 0
        grads = [0, 0, 0]
        for t in reversed(range(T)):
            layer = self.layers[t]
            dx, dh = layer.backward(dhs[:, t, :] + dh)
            dxs[:, t, :] = dx

            for i, grad in enumerate(layer.grads):
                grads[i] += grad
        
        for i, grad in enumerate(grads):
            self.grads[i][...] = grad
        
        self.dh = dh

        return dxs