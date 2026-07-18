import numpy as np

# Time Affine layer。
class TimeAffine:
    # TimeAffine layer の初期化。
    # param: self: TimeAffine layer。
    # param: W: 重み。(input_size, output_size) の形状を持つ。
    # param: b: バイアス。(output_size,) の形状を持つ。
    def __init__(
        self,
        W: np.ndarray[np.float32],
        b: np.ndarray[np.float32]
    ):
        self.params = [W, b]
        self.grads = [np.zeros_like(W), np.zeros_like(b)]
        self.x = None
    
    # 順伝播。
    # param: self: TimeAffine layer。
    # param: x: 入力データ。(batch_size, time_steps, input_size) の形状を持つ。
    # return: out: 出力データ。(batch_size, time_steps, output_size ) の形状を持つ。
    def forward(
        self,
        x: np.ndarray[np.float32]
    ) -> np.ndarray[np.float32]:
        N, T, D = x.shape
        W, b = self.params

        # (N, T, D) -> (N*T, D)
        rx = x.reshape(N*T, -1)

        # (N*T, output_size)
        out = np.dot(rx, W) + b
        self.x = x

        # (N*T, output_size) -> (N, T, output_size)
        return out.reshape(N, T, -1)
    
    # 逆伝播。
    # param: self: TimeAffine layer。
    # param: dout: 出力データの勾配。(batch_size, time_steps, output_size) の形状を持つ。
    # return: dx: 入力データの勾配。(batch_size, time_steps, input_size) の形状を持つ。
    def backward(
        self,
        dout: np.ndarray[np.float32]
    ) -> np.ndarray[np.float32]:
        x = self.x
        N, T, D = x.shape
        W, b = self.params

        # (N, T, output_size) -> (N*T, output_size)
        dout = dout.reshape(N*T, -1)
        
        # (N, T, D) -> (N*T, D)
        rx = x.reshape(N*T, -1)

        db = np.sum(dout, axis=0)
        dW = np.dot(rx.T, dout)
        dx = np.dot(dout, W.T)
        dx = dx.reshape(*x.shape)  # (N*T, D) -> (N, T, D)

        self.grads[0][...] = dW
        self.grads[1][...] = db

        return dx