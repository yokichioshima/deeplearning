import numpy as np
import sys
sys.path.append('..')
from layers.time_affine import TimeAffine
from layers.time_embedding import TimeEmbedding
from layers.time_lstm import TimeLSTM
from layers.time_rnn import TimeRNN
from layers.time_softmax_with_loss import TimeSoftmaxWithLoss
from models.base_model import BaseModel

class Rnnlm(BaseModel):
    # Rnnlm 初期化。
    # param: self: Rnnlm。
    # param: vocab_size: 単語数。
    # param: wordvec_size: 単語ベクトルのサイズ。
    # param: hidden_size: 隠れ層入力データサイズ。
    def __init__(
        self, 
        vocab_size: np.int32 =10000, 
        wordvec_size: np.int32 =100, 
        hidden_size: np.int32 =100
    ):
        V, D, H = vocab_size, wordvec_size, hidden_size
        rn = np.random.randn

        # 重みの初期化
        embed_W = (rn(V, D) / 100).astype('f')
        lstm_Wx = (rn(D, 4 * H) / np.sqrt(D)).astype('f')
        lstm_Wh = (rn(H, 4 * H) / np.sqrt(H)).astype('f')
        lstm_b = np.zeros(4 * H).astype('f')
        affine_W = (rn(H, V) / np.sqrt(H)).astype('f')
        affine_b = np.zeros(V).astype('f')

        # layer の生成
        self.layers = [
            TimeEmbedding(embed_W),
            TimeLSTM(lstm_Wx, lstm_Wh, lstm_b, stateful=True),
            TimeAffine(affine_W, affine_b)
        ]
        self.loss_layer = TimeSoftmaxWithLoss()
        self.lstm_layer = self.layers[1]

        # 全ての重みと勾配をリストにまとめる
        self.params, self.grads = [], []
        for layer in self.layers:
            self.params += layer.params
            self.grads += layer.grads
    
    # 予想。
    # param: self: Rnnlm。
    # param: xs: 入力データ。(batch_size, time_steps, input_size) の形状を持つ。
    # return xs: 出力データ。(batch_size, time_steps, input_size) の形状を持つ。
    def predict(
        self, 
        xs: np.ndarray[np.int32]
    ) -> np.ndarray[np.int32]:
        for layer in self.layers:
            xs = layer.forward(xs)
        return xs
    
    # 順伝播。
    # param: self: Rnnlm。
    # param: xs: 入力データ。(batch_size, time_steps, input_size) の形状を持つ。
    # param: tx: 正解ラベル。(batch_size,) の形状を持つ。
    def forward(
        self, 
        xs: np.ndarray[np.int32], 
        ts: np.ndarray[np.int32]
    ) -> np.float32:
        score = self.predict(xs)
        loss = self.loss_layer.forward(score, ts)
        return loss
    
    # 逆伝播。
    # param: self: Rnnlm。
    # param: dout: 上流からの勾配。
    # return: dout: 入力データの勾配。
    def backward(
        self, 
        dout: np.ndarray[np.float32]=1
    ) -> np.ndarray[np.float32]:
        dout = self.loss_layer.backward(dout)
        for layer in reversed(self.layers):
            dout = layer.backward(dout)
        return dout
    
    # 状態をリセットします。
    # param: self: Rnnlm。
    def reset_state(self):
        self.lstm_layer.reset_state()
