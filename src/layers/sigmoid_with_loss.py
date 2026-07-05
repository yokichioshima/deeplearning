import numpy as np
from common.functions import cross_entropy_error

class SigmoidWithLoss:
    def __init__(self):
        self.params, self.grads = [], []
        self.loss = None
        self.y = None # sigmoid の出力
        self.t = None # 教師ラベル
    
    # 順伝播。
    # param: self: SigmoidWithLoss。
    # param: x: 入力データ。
    # param: t: 教師ラベル。
    # return: loss: 損失値。
    def forward(
        self, 
        x: np.ndarray[np.float32],
        t: np.ndarray[np.int32]
    ) -> np.float32:
        self.t = t
        self.y = 1 / (1 + np.exp(-x))

        self.loss = cross_entropy_error(np.c_[1 - self.y, self.y], self.t)
        
        return self.loss
    
    # 逆伝播。
    # param: self: SigmoidWithLoss。
    # param: dout: 出力の勾配。
    # return: dx: 入力の勾配。
    def backward(
        self, 
        dout: np.ndarray[np.float32] = 1
    ) -> np.ndarray[np.float32]:
        batch_size = self.t.shape[0]

        dx = (self.y - self.t) * dout / batch_size
        
        return dx