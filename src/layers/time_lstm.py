import numpy as np
from layers.lstm import LSTM

class TimeLSTM:
    # TimeLSTM layer の初期化。
    # param: self: TimeLSTM layer。
    # param: Wx: 入力 x に対する重み。(input_size, hidden_size) の形状を持つ。
    # param: Wh: 前の時刻の隠れ状態 h に対する重み。(hidden_size, hidden_size) の形状を持つ。
    # param: b: バイアス。(hidden_size,) の形状を持つ。
    # param: stateful: True の場合、隠れ状態を保持する。
    def __init__(
        self, 
        Wx: np.ndarray[np.float32], 
        Wh: np.ndarray[np.float32], 
        b: np.ndarray[np.float32], 
        stateful: bool =False
    ):
        self.params = [Wx, Wh, b]
        self.grads = [np.zeros_like(Wx), np.zeros_like(Wh), np.zeros_like(b)]
        self.layers = None

        self.h, self.c = None, None
        self.dh = None
        self.stateful = stateful
    
    # 順伝播。
    # param: self: TimeLSTM layer。
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
        if not self.stateful or self.c is None:
            self.c = np.zeros((N, H), dtype='f')
        
        for t in range(T):
            layer = LSTM(*self.params)
            self.h, self.c = layer.forward(xs[:, t, :], self.h, self.c)
            hs[:, t, :] = self.h

            self.layers.append(layer)
        
        return hs
    
    # 逆伝播。
    # param: self: TimeLSTM layer。
    # param: dhs: 隠れ状態の勾配。(batch_size, time_steps, hidden_size) の形状を持つ。
    # param: dxs: 入力データの勾配。(batch_size, time_steps, input_size) の形状を持つ。
    def backward(
        self,
        dhs: np.ndarray[np.float32]
    ) -> np.ndarray[np.float32]:
        Wx, Wh, b = self.params
        N, T, H = dhs.shape
        D = Wx.shape[0]

        dxs = np.empty((N, T, D), dtype='f')
        dh, dc = 0, 0

        grads = [0, 0, 0]
        for t in reversed(range(T)):
            layer = self.layers[t]
            dx, dh, dc = layer.backward(dhs[:, t, :] + dh, dc)
            dxs[:, t, :] = dx
            for i, grad in enumerate(layer.grads):
                grads[i] += grad
        
        for i, grad in enumerate(grads):
            self.grads[i][...] = grad
        
        self.dh = dh
        return dxs
    
    # 状態を設定します。
    # param: self: TimeLSTM layer。
    # param: h: 隠れ状態。(batch_size, time_steps, hidden_size) の形状を持つ。
    # param: c: 記憶セル。(batch_size, time_steps, hidden_size) の形状を持つ。
    def set_state(
        self, 
        h: np.ndarray[np.float32], 
        c: np.ndarray[np.float32] =None
    ):
        self.h, self.c = h, c
    
    # 状態をリセットします。
    # param: self: TimeLSTM layer。
    def reset_state(self):
        self.h, self.c = None, None
