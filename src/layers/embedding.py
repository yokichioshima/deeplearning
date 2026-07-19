import numpy as np

class Embedding:
    # Embedding layer の初期化。
    # param: self: Embedding layer。
    # param: W: 埋め込み行列。(vocab_size, wordvec_size) の形状を持つ。
    def __init__(
        self, 
        W: np.ndarray[np.float32]
    ):
        self.params = [W]
        self.grads = [np.zeros_like(W)]
        self.idx = None
    
    # 順伝播。
    # param: self: Embedding layer。
    # param: idx: word_id の配列。(batch_size,) の形状を持つ。
    # return: out: 出力データ。(batch_size, wordvec_size) の形状を持つ。
    def forward(
        self, 
        idx: np.ndarray[np.int32]
    ) -> np.ndarray[np.float32]:
        W, = self.params
        self.idx = idx
        out = W[idx]
        return out
    
    # 逆伝播。
    # param: self: Embedding layer。
    # param: dout: 出力データの勾配。(batch_size, wordvec_size) の形状を持つ。
    # return: None。
    def backward(
        self,
        dout: np.ndarray[np.float32]
    ) -> None:
        dW, = self.grads
        dW[...] = 0
        np.add.at(dW, self.idx, dout)
        return None