import numpy as np

class Dropout:
    # Dropout layer の初期化。
    # param: self: Dropout layer。
    # param: dropout_ratio: 順伝播、逆伝播で 0 にする要素の個数の割合。
    def __init__(
        self, 
        dropout_ratio: np.float32 =0.5):
        self.params, self.grads = [], []
        self.dropout_ratio = dropout_ratio
        self.mask = None
    
    # 順伝播。
    # param: self: Dropout layer。
    # param: x: 入力データ。
    # param: train_flg: Ture: 訓練を行う。False: 訓練を行わない。
    def forward(
        self, 
        x: np.ndarray[np.float32], 
        train_flg: bool =True
    ) -> np.ndarray[np.float32]:
        if train_flg:
            self.mask = np.random.rand(*x.shape) > self.dropout_ratio
            return x * self.mask
        else:
            return x * (1.0 - self.dropout_ratio)
    
    # 逆伝播。
    # param: self: Dropout layer。
    # param: dout: 上流からの勾配。
    # return: 勾配。
    def backward(
        self, 
        dout
    ) -> np.ndarray[np.float32]:
        return dout * self.mask