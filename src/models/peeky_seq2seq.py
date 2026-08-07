import sys
sys.path.append('..')
import numpy as np
from layers.time_affine import TimeAffine
from layers.time_dropout import TimeDropout
from layers.time_embedding import TimeEmbedding
from layers.time_lstm import TimeLSTM
from layers.time_rnn import TimeRNN
from layers.time_softmax_with_loss import TimeSoftmaxWithLoss
from models.seq2seq import Seq2seq, Encoder

class PeekyDecoder:
    # PeekyDecoder layer の初期化。
    # param: vocab_size: 語彙数。
    # param: wordvec_size: 単語ベクトルの次元数。
    # param: hidden_size: 隠れ状態の次元数。
    def __init__(
        self,
        vocab_size: np.int32,
        wordvec_size: np.int32,
        hidden_size: np.int32,
    ):
        V, D, H = vocab_size, wordvec_size, hidden_size
        rn = np.random.randn

        # 重みの初期化
        embed_W = (rn(V, D) / 100).astype('f')
        lstm_Wx = (rn(D + H, 4 * H) / np.sqrt(D + H)).astype('f')
        lstm_Wh = (rn(H, 4 * H) / np.sqrt(H)).astype('f')
        lstm_b = np.zeros(4 * H).astype('f')
        affine_W = (rn(H + H, V) / np.sqrt(H + H)).astype('f')
        affine_b = np.zeros(V).astype('f')

        # レイヤの生成
        self.embed = TimeEmbedding(embed_W)
        self.lstm = TimeLSTM(lstm_Wx, lstm_Wh, lstm_b, stateful=True)
        self.affine = TimeAffine(affine_W, affine_b)
        
        self.params, self.grads = [], []
        for layer in (self.embed, self.lstm, self.affine):
            self.params += layer.params
            self.grads += layer.grads
        self.cache = None
    
    # 順伝播。
    # param: self: PeekyDecoder layer。
    # param: xs: 入力データ。(batch_size, time_steps) の形状を持つ。
    # param: h: エンコーダの隠れ状態。(batch_size, hidden_size) の形状を持つ。
    # return: score: 出力スコア。(batch_size, time_steps, vocab_size) の形状を持つ。
    def forward(
        self, 
        xs: np.ndarray[np.int32], 
        h: np.ndarray[np.float32]
    ) -> np.ndarray[np.float32]:
        N, T = xs.shape
        N, H = h.shape

        self.lstm.set_state(h)

        out = self.embed.forward(xs)
        hs = np.repeat(h, T, axis=0).reshape(N, T, H)
        out = np.concatenate((hs, out), axis=2)

        out = self.lstm.forward(out)
        out = np.concatenate((hs, out), axis=2)

        score = self.affine.forward(out)
        self.cache = H
        return score
    
    # 逆伝播。
    # param: self: PeekyDecoder layer。
    # param: dscore: 出力スコアの勾配。(batch_size, time_steps, vocab_size) の形状を持つ。
    # return: dh: エンコーダの隠れ状態の勾配。(batch_size, hidden_size) の形状を持つ。
    def backward(
        self, 
        dscore: np.ndarray[np.float32]
    ) -> np.ndarray[np.float32]:
        H = self.cache

        dout = self.affine.backward(dscore)
        dhs0, dout = dout[:, :, :H], dout[:, :, H:]
        dout = self.lstm.backward(dout)
        dembed, dhs1 = dout[:, :, H:], dout[:, :, :H]
        self.embed.backward(dembed)
        dhs = dhs0 + dhs1
        dh = self.lstm.dh + np.sum(dhs, axis=1)
        return dh
    
    # 文章生成。
    # param: self: PeekyDecoder layer。
    # param: h: エンコーダの隠れ状態。(batch_size, hidden_size) の形状を持つ。
    # param: start_id: 文章生成の開始文字の ID。
    # param: sample_size: 生成する文章の長さ。
    # return: sampled: 生成された文章の文字 ID の配列。(sample_size,) の形状を持つ。
    def generate(
        self, 
        h: np.ndarray[np.float32], 
        start_id: np.int32, 
        sample_size: np.int32
    ) -> np.ndarray[np.int32]:
        sampled = []
        char_id = start_id
        self.lstm.set_state(h)

        H = h.reshape[1]
        peeky_h = h.reshape(1, 1, H)
        for _ in range(sample_size):
            x = np.array([char_id]).reshape((1, 1))
            out = self.embed.forward(x)

            out = np.concatenate((peeky_h, out), axis=2)
            out = self.lstm.forward(out)
            out = np.concatenate((peeky_h, out), axis=2)
            score = self.affine.forward(out)
            
            char_id = np.argmax(score.flatten())
            sampled.append(int(char_id))

        return np.array(sampled)
    
class PeekySeq2seq(Seq2seq):
    # PeekySeq2seq layer の初期化。
    # param: vocab_size: 語彙数。
    # param: wordvec_size: 単語ベクトルの次元数。
    # param: hidden_size: 隠れ状態の次元数。
    def __init__(
        self,
        vocab_size: np.int32,
        wordvec_size: np.int32,
        hidden_size: np.int32
    ):
        V, D, H = vocab_size, wordvec_size, hidden_size
        self.encoder = Encoder(V, D, H)
        self.decoder = PeekyDecoder(V, D, H)
        self.softmax = TimeSoftmaxWithLoss()

        self.params = self.encoder.params + self.decoder.params
        self.grads = self.encoder.grads + self.decoder.grads


