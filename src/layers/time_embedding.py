import numpy as np
from layers.embedding import Embedding

# TimeEmbedding layer。
class TimeEmbedding:
    # TimeEmbedding layer の初期化。
    # param: self: TimeEmbedding layer。
    # param: W: 埋め込み行列。(vocab_size, embedding_size) の形状を持つ。
    def __init__(
        self, 
        W: np.ndarray[np.float32]
    ):
        self.params = [W]
        self.grads = [np.zeros_like(W)]
        self.layers = None
        self.W = W
    
    # 順伝播。
    # param: self: TimeEmbedding layer。
    # param: xs: 入力データ。(batch_size, time_steps) の形状を持つ。
    # return: out: 出力データ。(batch_size, time_steps, embedding_size) の形状を持つ。 
    def forward(
        self, 
        xs: np.ndarray[np.float32]
    ) -> np.ndarray[np.float32]:
        N, T = xs.shape
        V, D = self.W.shape

        out = np.empty((N, T, D), dtype='f')
        self.layers = []

        for t in range(T):
            layer = Embedding(self.W)
            out[:, t, :] = layer.forward(xs[:, t])
            self.layers.append(layer)
        
        return out
    
    # 逆伝播。
    # param: self: TimeEmbedding layer。
    # param: dout: 出力データの勾配。(batch_size, time_steps, embedding_size) の形状を持つ。
    # return: None。
    def backward(
        self,
        dout: np.ndarray[np.float32]
    ) -> None:
        N, T, D = dout.shape

        grad = 0
        for t in range(T):
            layer = self.layers[t]
            layer.backward(dout[:, t, :])
            grad += layer.grads[0]
        
        self.grads[0][...] = grad
        return None