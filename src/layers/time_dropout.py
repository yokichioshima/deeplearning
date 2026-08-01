import numpy as np

class TimeDropout:
    # TimeDropout layer の初期化。
    # param: self: TimeDropout layer。
    # param: dropout_ratio: 順伝播、逆伝播で 0 にする要素の個数の割合。
    def __init__(self, dropout_ratio=0.5):
        self.params, self.grads = [], []
        self.dropout_ratio = dropout_ratio
        self.mask = None
        self.train_flg = True
    
    # 順伝播。
    # param: self: TimeDropout layer。
    # param: xs: 入力データ。
    # return 出力データ。
    def forward(
        self, 
        xs: np.ndarray[np.float32]
    ) -> np.ndarray[np.float32]:
        if self.train_flg:
            flg = np.random.rand(*xs.shape) > self.dropout_ratio
            scale = 1 / (1.0 - self.dropout_ratio)
            self.mask = flg.astype(np.float32) * scale

            return xs * self.mask
        else:
            return xs

    # 逆伝播。
    # param: self: TimeDropout layer。
    # param: dout: 上流からの勾配。
    # return: 勾配。
    def backward(
        self, 
        dout: np.ndarray[np.float32]
    ) -> np.ndarray[np.float32]:
        return dout * self.mask
    