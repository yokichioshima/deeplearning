import numpy as np
import sys
sys.path.append('..')
from layers.time_affine import TimeAffine
from layers.time_dropout import TimeDropout
from layers.time_embedding import TimeEmbedding
from layers.time_lstm import TimeLSTM
from models.base_model import BaseModel

class BetterRnnlm(BaseModel):
    '''
        LSTM layer を 2 層利用し、各層に Dropout を使う model。
        [1] で提案されたモデルをベースとし、weight tying [2][3] を利用

        [1] Recurrent Neural Network Regularization (https://arxiv.org/abs/1409.2329)
        [2] Using the Output Embedding to Improve Language Models (https://arxiv.org/abs/1608.05859)
        [3] Tying Word Vectors and Word Classifiers (https://arxiv.org/pdf/1611.01462.pdf)
    '''

    # BetterRnnlm の初期化。
    # self: BetterRnnlm。
    # vocab_size: 語彙数。
    # wordvec_size: 単語ベクトルサイズ。
    # hidden_size: 隠れ層入力データサイズ。
    # dropout_ratio: 順伝播、逆伝播で 0 にする要素の個数の割合。
    def __init__(
        self, 
        vocab_size: np.int32 =10000, 
        wordvec_size: np.int32 =650,
        hidden_size: np.int32 =650, 
        dropout_ratio: np.float32 =0.5
    ):
        V, D, H = vocab_size, wordvec_size, hidden_size
        rn = np.random.randn

        embed_W = (rn(V, D) / 100).astype('f')
        lstm_Wx1 = (rn(D, 4 * H) / np.sqrt(D)).astype('f')
        lstm_Wh1 = (rn(H, 4 * H) / np.sqrt(D)).astype('f')
        lstm_b1 = np.zeros(4 * H).astype('f')
        lstm_Wx2 = (rn(H, 4 * H) / np.sqrt(H)).astype('f')
        lstm_Wh2 = (rn(H, 4 * H) / np.sqrt(H)).astype('f')
        lstm_b2 = np.zeros(4 * H).astype('f')
        affine_b = np.zeros(V).astype('f')

        self.layers = [
            TimeEmbedding(embed_W),
            TimeDropout(dropout_ratio),
            TimeLSTM(lstm_Wx1, lstm_Wh1, lstm_b1, stateful=True),
            TimeDropout(dropout_ratio),
            TimeLSTM(lstm_Wx2, lstm_Wh2, lstm_b2, stateful=True),
            TimeDropout(dropout_ratio),
            TimeAffine(embed_W.T, affine_b) # 重み共有
        ]
        self.loss_layer = TimeSoftmaxWithLoss()
        self.lstm_layers = [self.layers[2], self.layers[4]]
        self.drop_layers = [self.layers[1], self.layers[3], self.layers[5]]

        self.params, self.grads = [], []
        for layer in self.layers:
            self.params += layer.params
            self.grads += layer.grads
    
    # 予想。
    # param: self: BetterRnnlm。
    # param: xs: 入力データ。(batch_size, time_steps, input_size) の形状を持つ。
    # param: train_flg: Ture: 訓練を行う。False: 訓練を行わない。
    # return xs: 出力データ。(batch_size, time_steps, input_size) の形状を持つ。
    def predict(
        self, 
        xs: np.ndarray[np.float32], 
        train_flg: bool =False
    ) -> np.ndarray[np.float32]:
        for layer in self.drop_layers:
            layer.train_flg = train_flg
        for layer in self.layers:
            xs = layer.forward(xs)
        return xs
    
    # 順伝播。
    # self: BetterRnnlm。
    # xs: 入力データ。(batch_size, time_steps, input_size) の形状を持つ。
    # ts: 正解ラベル。(batch_size,) の形状を持つ。
    def forward(
        self,
        xs: np.ndarray[np.float32], 
        ts: np.ndarray[np.float32], 
        train_flg: bool =True
    ) -> np.ndarray[np.float32]:
        score = self.predict(xs, train_flg)
        loss = self.loss_layer.forward(score, ts)
        return loss
    
    # 逆伝播。
    # param: self: BetterRnnlm。
    # param: dout: 上流からの勾配。
    # return: dout: 入力データの勾配。
    def backward(
        self, 
        dout: np.ndarray[np.float32] =1
    ) -> np.ndarray[np.float32]:
        dout = self.loss_layer.backward(dout)
        for layer in reversed(self.layers):
            dout = layer.backward(dout)
        return dout
    
    # 状態をリセットします。
    # param: self: Rnnlm。
    def reset_state(self):
        for layer in self.lstm_layers:
            layer.reset_state()
