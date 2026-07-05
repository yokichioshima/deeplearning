import numpy as np
from layers.embedding import Embedding
from layers.negative_sampling_loss import NegativeSamplingLoss

class CBOW:
    # CBOW モデルの初期化
    # param: vocab_size: 語彙数。
    # param: hidden_size: 隠れ層のニューロン数。
    # param: window_size: ウィンドウサイズ。
    # param: corpus: コーパス(単語 ID リスト)。
    def __init__(
        self, 
        vocab_size: int, 
        hidden_size: int, 
        window_size: int, 
        corpus: np.ndarray[np.int32]
    ):
        V, H = vocab_size, hidden_size

        # 重みの初期化
        W_in = 0.01 * np.random.randn(V, H).astype('f')
        W_out = 0.01 * np.random.randn(V, H).astype('f')

        # layer の生成
        self.in_layers = []
        for i in range(2 * window_size):
            layer = Embedding(W_in)
            self.in_layers.append(layer)
        self.ns_loss = NegativeSamplingLoss(W_out, corpus, power=0.75, sample_size=5)

        # 全ての重みと勾配を配列にまとめる
        layers = self.in_layers + [self.ns_loss]
        self.params, self.grads = [], []
        for layer in layers:
            self.params += layer.params
            self.grads += layer.grads

        # 単語の分散表現を保持
        self.word_vecs = W_in

    # 順伝播
    # param: self: CBOW model。
    # param: contexts: コンテキスト。2 次元配列で、各行がコンテキストの単語 ID のリスト。
    # param: target: ターゲット。1 次元配列で、各行がターゲットの単語 ID。
    # return: loss: 損失値。
    def forward(
        self, 
        contexts: np.ndarray[np.int32], 
        target: np.ndarray[np.int32]
    ) -> float:
        h = 0
        for i, layer in enumerate(self.in_layers):
            h += layer.forward(contexts[:, i])
        h *= 1 / len(self.in_layers)

        loss = self.ns_loss.forward(h, target)
        return loss
    
    # 逆伝播
    # param: self: CBOW model。
    # param: dout: 逆伝播の勾配。
    # return: None。
    def backward(self, dout: float = 1):
        dout = self.ns_loss.backward(dout)
        dout *= 1 / len(self.in_layers)
        for layer in self.in_layers:
            layer.backward(dout)
        return None