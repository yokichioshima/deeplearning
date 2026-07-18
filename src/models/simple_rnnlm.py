import sys
sys.path.append('..')
import numpy as np
from common.functions import softmax
from layers.time_embedding import TimeEmbedding
from layers.time_rnn import TimeRNN
from layers.time_affine import TimeAffine
from layers.time_softmax_with_loss import TimeSoftmaxWithLoss

class SimpleRnnlm:
    # SimpleRnnlm model の初期化。
    # param: self: SimpleRnnlm model。
    # param: vocab_size: 単語数。
    # param: wordvec_size: 単語ベクトルのサイズ。
    # param: hidden_size: 隠れ層入力データサイズ。
    def __init__(
        self,
        vocab_size: np.int32,
        wordvec_size: np.int32,
        hidden_size: np.int32
    ):
        V, D, H = vocab_size, wordvec_size, hidden_size
        rn = np.random.randn

        # 重みの初期化
        embed_W = (rn(V, D) / 100).astype('f')
        rnn_Wx = (rn(D, H) / np.sqrt(D)).astype('f')
        rnn_Wh = (rn(H, H) / np.sqrt(H)).astype('f')
        rnn_b = np.zeros(H).astype('f')
        affine_W = (rn(H, V) / np.sqrt(H)).astype('f')
        affine_b = np.zeros(V).astype('f')

        # layer の生成
        self.layers = [
            TimeEmbedding(embed_W),
            TimeRNN(rnn_Wx, rnn_Wh, rnn_b, stateful=True),
            TimeAffine(affine_W, affine_b)
        ]
        self.loss_layer = TimeSoftmaxWithLoss()
        self.rnn_layer = self.layers[1]

        # すべての重みと勾配をリストにまとめる
        self.params, self.grads = [], []
        for layer in self.layers:
            self.params += layer.params
            self.grads += layer.grads
    
    # 順伝播。
    # param: self: SimpleRnnlm model。
    # param: xs: word_id の配列。
    # param: ts: 正解ラベル。
    def forward(
        self, 
        xs: np.ndarray[np.int32], 
        ts: np.ndarray[np.int32]
    ) -> np.float32:
        for layer in self.layers:
            xs = layer.forward(xs)
        loss = self.loss_layer.forward(xs, ts)
        return loss
    
    # 逆伝播。
    # param: self: SimpleRnnlm model。
    # param: dout: 上流からの勾配。
    # return dout: 入力データの勾配
    def backward(
        self,
        dout: np.float32 = 1
    ) -> np.float32:
        dout = self.loss_layer.backward(dout)
        for layer in reversed(self.layers):
            dout = layer.backward(dout)
        return dout
    
    def reset_state(self):
        self.rnn_layer.reset_state()