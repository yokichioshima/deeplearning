import sys
sys.path.append('..')
import numpy as np
from layers.time_affine import TimeAffine
from layers.time_attention import TimeAttention
from layers.time_embedding import TimeEmbedding
from layers.time_lstm import TimeLSTM
from layers.time_softmax_with_loss import TimeSoftmaxWithLoss
from models.seq2seq import Encoder, Seq2seq

class AttentionEncoder(Encoder):
    # 順伝播。
    # param: self: AttentionEncoder。
    # param: xs: 入力データ。(batch_size, time_steps) の形状を持つ。
    # return: 隠れ状態の配列。(batch_size, time_steps, hidden_size) の形状を持つ。
    def forward(
        self, 
        xs: np.ndarray[np.float32]
    ) -> np.ndarray[np.float32]:
        xs = self.embed.forward(xs)
        hs = self.lstm.forward(xs)
        return hs
    
    # 逆伝播。
    # param: self: AttentionEncoder。
    # param: dhs: 隠れ状態の配列の勾配。(batch_size, time_steps, hidden_size) の形状を持つ。
    # return: dout: None。
    def backward(
        self,
        dhs: np.ndarray[np.float32]
    ) -> None:
        dout = self.lstm.backward(dhs)
        dout = self.embed.backward(dout)
        return dout
    
class AttentionDecoder:
    # 初期化。
    # param: self: AttentionDecoder。
    # param: vocab_size: 語彙数。
    # param: wordvec_size: 単語ベクトルのサイズ。
    # param: hidden_size: 隠れ層入力データサイズ。
    def __init__(
        self,
        vocab_size,
        wordvec_size,
        hidden_size
    ):
        V, D, H = vocab_size, wordvec_size, hidden_size
        rn = np.random.randn

        embed_W = (rn(V, D) / 100).astype('f')
        lstm_Wx = (rn(D, 4 * H) / np.sqrt(D)).astype('f')
        lstm_Wh = (rn(H, 4 * H) / np.sqrt(H)).astype('f')
        lstm_b = np.zeros(4 * H).astype('g')
        affine_W = (rn(2*H, V) / np.sqrt(2*H)).astype('f')
        affine_b = np.zeros(V).astype('f')

        self.embed = TimeEmbedding(embed_W)
        self.lstm = TimeLSTM(lstm_Wx, lstm_Wh, lstm_b, stateful=True)
        self.attention = TimeAttention()
        self.affine = TimeAffine(affine_W, affine_b)
        layers = [self.embed, self.lstm, self.attention, self.affine]

        self.params, self.grads = [], []
        for layer in layers:
            self.params += layer.params
            self.grads += layer.grads
    
    # 順伝播。
    # param: self: AttentionDecoder。
    # param: xs: 入力データ。(batch_size, time_steps, hidden_size) の形状を持つ。
    # param: enc_hs: Encoder からの隠れ状態の配列。(batch_size, time_steps, hidden_size) の形状を持つ。
    # return: score: スコア。(batch_size, time_steps, output_size) の形状を持つ。
    def forward(
        self, 
        xs: np.ndarray[np.float32], 
        enc_hs: np.ndarray[np.float32]
    ) -> np.ndarray[np.float32]:
        h = enc_hs[:, -1]
        self.lstm.set_state(h)

        out = self.embed.forward(xs)
        dec_hs = self.lstm.forward(out)
        c = self.attention.forward(enc_hs, dec_hs)
        out = np.concatenate((c, dec_hs), axis=2)
        score = self.affine.forward(out)

        return score
    
    # 逆伝播。
    # param: self: AttentionDecoder。
    # param: dscore: スコアの勾配。(batch_size, time_steps, output_size) の形状を持つ。
    # return: denc_hs: Encoder からの隠れ状態の配列の勾配。(batch_size, time_steps, hidden_size) の形状を持つ。
    def backward(
        self,
        dscore: np.ndarray[np.float32]
    ) -> np.ndarray[np.float32]:
        dout = self.affine.backward(dscore)
        N, T, H2 = dout.shape
        H = H2 // 2

        dc, ddec_hs0 = dout[:, :, :H], dout[:, :, H:]
        denc_hs, ddec_hs1 = self.attention.backward(dc)
        ddec_hs = ddec_hs0 + ddec_hs1
        dout = self.lstm.backward(ddec_hs)
        dh = self.lstm.dh
        denc_hs[:, -1] += dh
        self.embed.backward(dout)

        return denc_hs

    # 文章生成。
    # param: self: AttentionDecoder。
    # param: enc_hs: Encoder からの隠れ状態の配列。(batch_size, time_steps, hidden_size) の形状を持つ。
    # param: start_id: 開始する単語の ID 。
    # param: sample_size: 文章の単語数。
    # return: 生成された単語の ID の配列。
    def generate(
        self,
        enc_hs: np.ndarray[np.float32],
        start_id: np.int32,
        sample_size: np.int32
    ) -> np.ndarray[np.int32]:
        sampled = []
        sample_id = start_id
        h = enc_hs[:, -1]
        self.lstm.set_state(h)

        for _ in range(sample_size):
            x = np.array([sample_id]).reshape((1, 1))

            out = self.embed.forward(x)
            dex_hs = self.lstm.forward(out)
            c = self.attention.forward(enc_hs, dec_hs)
            out = np.concatenate((c, dec_hs), axis=2)
            score = self.affine.forward(out)

            sample_id = np.argmax(score.flatten())
            sampled.append(sample_id)
        
        return sampled

class AttentionSeq2seq(Seq2seq):
    # 初期化。
    # param: self: AttentionSeq2seq。
    # param: vocab_size: 語彙数。
    # param: wordvec_size: 単語ベクトルのサイズ。
    # param: hidden_size: 隠れ層の入力データサイズ。
    def __init__(
        self, 
        vocab_size: np.int32, 
        wordvec_size: np.int32, 
        hidden_size: np.int32
    ):
        args = vocab_size, wordvec_size, hidden_size
        self.encoder = AttentionEncoder(*args)
        self.decoder = AttentionDecoder(*args)
        self.softmax = TimeSoftmaxWithLoss()

        self.params = self.encoder.params + self.decoder.params
        self.grads = self.encoder.grads + self.decoder.grads
    