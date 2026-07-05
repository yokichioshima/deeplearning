import numpy as np
from layers.embedding import Embedding
from layers.matmul import MatMul
from layers.softmax_with_loss import SoftMaxWithLoss

class SimpleCBOW:
    # SimpleCBOW の初期化。
    # param: self: SimpleCBOW。
    # param: vocab_size: 語彙数。
    # param: hidden_size: 隠れ層のニューロン数。
    def __init__(self, vocab_size: int, hidden_size: int):
        V, H = vocab_size, hidden_size

        # 重みの初期化
        W_in = 0.01 * np.random.randn(V, H).astype('f')
        W_out = 0.01 * np.random.randn(H, V).astype('f')

        # layer の生成
        self.in_layer0 = Embedding(W_in)
        self.in_layer1 = Embedding(W_in)
        self.out_layer = MatMul(W_out)
        self.loss_layer = SoftMaxWithLoss()

        # 全ての重みと勾配をリストにまとめる
        layers = [self.in_layer0, self.in_layer1, self.out_layer]
        self.params, self.grads = [], []
        for layer in layers:
            self.params += layer.params
            self.grads += layer.grads
        
        # 単語の分散表現を保持
        self.word_vecs = W_in
    
    # 順伝播。
    # param: self: SimpleCBOW。
    # param: contexts: コンテキスト。2 次元配列で、各行がコンテキストの単語 ID のリスト。
    # param: target: ターゲット。1 次元配列で、各行がターゲットの単語 ID。
    # return: loss: 損失値。
    def forward(self, contexts: np.ndarray[np.int32], target: np.ndarray[np.int32]) -> float:
        h0 = self.in_layer0.forward(contexts[:, 0])
        h1 = self.in_layer1.forward(contexts[:, 1])
        h = (h0 + h1) * 0.5
        score = self.out_layer.forward(h)
        loss = self.loss_layer.forward(score, target)
        return loss
    
    # 逆伝播。
    # param: self: SimpleCBOW。
    # param: dout: 逆伝播の勾配。
    # return: None。
    def backward(self, dout=1):
        ds = self.loss_layer.backward(dout)
        da = self.out_layer.backward(ds)
        da *= 0.5
        self.in_layer0.backward(da)
        self.in_layer1.backward(da)
        return None
    


