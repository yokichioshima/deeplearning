import numpy as np
from common.functions import softmax

class TimeSoftmaxWithLoss:
    # TimeSoftmaxWithLoss layer の初期化。
    # param: self: TimeSoftmaxWithLoss layer。
    def __init__(self):
        self.params, self.grads = [], []
        self.cache = None
        self.ignore_label = -1
    
    # 順伝播。
    # param: self: TimeSoftmaxWithLoss layer。
    # param: xs: 入力データ。(batch_size, time_steps, vocab_size) の形状を持つ。
    # param: ts: 教師ラベル。(batch_size, time_steps) の形状を持つ。
    # return: loss: 損失値。
    def forward(
        self,
        xs: np.ndarray[np.float32],
        ts: np.ndarray[np.float32]
    ) -> np.float32:
        N, T, V = xs.shape

        if ts.ndim == 3:
            # 教師ラベルが one-hot ベクトルの場合
            ts = ts.argmax(axis=2)

        mask = (ts != self.ignore_label)

        xs = xs.reshape(N*T, V)
        ts = ts.reshape(N*T)
        mask = mask.reshape(N*T)

        ys = softmax(xs)
        ls = np.log(ys[np.arange(N*T), ts])
        ls *= mask # ignore_label の場合は損失を計算しない
        loss = -np.sum(ls) / mask.sum()

        self.cache = (ts, ys, mask, (N, T, V))
        return loss

    # 逆伝播。
    # param: self: TimeSoftmaxWithLoss layer。
    # param: dout: 上流からの勾配。
    # return: dx: 入力データの勾配。
    def backward(
        self, 
        dout: np.float32 = 1
    ) -> np.ndarray[np.float32]:
        ts, ys, mask, (N, T, V) = self.cache

        dx = ys
        dx[np.arange(N*T), ts] -= 1
        dx *= dout
        dx /= mask.sum()
        dx *= mask[:, np.newaxis] # ignore_label の場合は勾配を計算しない

        dx = dx.reshape((N, T, V))

        return dx