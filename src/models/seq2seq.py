import sys
sys.path.append('..')
import numpy as np
from layers.time_affine import TimeAffine
from layers.time_dropout import TimeDropout
from layers.time_embedding import TimeEmbedding
from layers.time_lstm import TimeLSTM
from layers.time_softmax_with_loss import TimeSoftmaxWithLoss
from models.base_model import BaseModel

class Encoder:
    # Encoder の初期化。
    # param: self: Encoder。
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
        lstm_Wx = (rn(D, 4 * H) / np.sqrt(D)).astype('f')
        lstm_Wh = (rn(H, 4 * H) / np.sqrt(H)).astype('f')
        lstm_b = np.zeros(4 * H).astype('f')

        # layer の生成
        self.embed = TimeEmbedding(embed_W)
        self.lstm = TimeLSTM(lstm_Wx, lstm_Wh, lstm_b, stateful=False)

        self.params = self.embed.params + self.lstm.params
        self.grads = self.embed.grads + self.lstm.grads
        self.hs = None
    
    # 順伝播。
    # param: self: Encoder。
    # param: xs: 入力データ。(batch_size, time_steps, hidden_size) の形状を持つ。
    # return xs: 出力データ。(batch_size, time_steps, hidden_size) の形状を持つ。
    def forward(self, xs: np.ndarray[np.int32]) -> np.ndarray[np.float32]:
        xs = self.embed.forward(xs)
        hs = self.lstm.forward(xs)
        self.hs = hs
        return hs[:, -1, :]
    
    # 逆伝播。
    # param: self: Encoder。
    # param: dh: 逆伝播の勾配。(batch_size, hidden_size) の形状を持つ。
    # return: dout: 逆伝播の勾配。(batch_size, time_steps) の形状を持つ。
    def backward(self, dh: np.ndarray[np.float32]) -> np.ndarray[np.float32]:
        dhs = np.zeros_like(self.hs)
        dhs[:, -1, :] = dh

        dout = self.lstm.backward(dhs)
        dout = self.embed.backward(dout)
        return dout

class Decoder:
    # Decoder の初期化。
    # param: self: Decoder。
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
        lstm_Wx = (rn(D, 4 * H) / np.sqrt(D)).astype('f')
        lstm_Wh = (rn(H, 4 * H) / np.sqrt(H)).astype('f')
        lstm_b = np.zeros(4 * H).astype('f')
        affine_W = (rn(H, V) / np.sqrt(H)).astype('f')
        affine_b = np.zeros(V).astype('f')

        # layer の生成
        self.embed = TimeEmbedding(embed_W)
        self.lstm = TimeLSTM(lstm_Wx, lstm_Wh, lstm_b, stateful=True)
        self.affine = TimeAffine(affine_W, affine_b)

        self.params, self.grads = [], []
        for layer in (self.embed, self.lstm, self.affine):
            self.params += layer.params
            self.grads += layer.grads
    
    # 順伝播。
    # param: self: Decoder。
    # param: xs: 入力データ。(batch_size, time_steps, hidden_size) の形状を持つ。
    # param: h: 隠れ層の初期値。(batch_size, hidden_size) の形状を持つ。
    # return: score: 出力データ。(batch_size, time_steps, vocab_size) の形状を持つ。
    def forward(
        self, 
        xs: np.ndarray[np.int32], 
        h: np.ndarray[np.float32]
    ) -> np.ndarray[np.float32]:
        self.lstm.set_state(h)

        out = self.embed.forward(xs)
        out = self.lstm.forward(out)
        score = self.affine.forward(out)
        return score
    
    # 逆伝播。
    # param: self: Decoder。
    # param: dscore: 逆伝播の勾配。(batch_size, time_steps, vocab_size) の形状を持つ。
    # return: dh: 逆伝播の勾配。(batch_size, hidden_size) の形状を持つ。
    def backward(
        self, 
        dscore: np.ndarray[np.float32]
    ) -> np.ndarray[np.float32]:
        dout = self.affine.backward(dscore)
        dout = self.lstm.backward(dout)
        dout = self.embed.backward(dout)
        dh = self.lstm.dh
        return dh
    
    # 文章生成。
    # param: self: Decoder。
    # param: h: 隠れ層の初期値。(batch_size, hidden_size) の形状を持つ。
    # param: start_id: 文章生成の開始単語ID。
    # param: sample_size: 文章生成の単語数。
    # return: sampled: 生成された文章の単語IDのリスト。
    def generate(
        self, 
        h: np.ndarray[np.float32], 
        start_id: np.int32, 
        sample_size: np.int32
    ) -> np.ndarray[np.int32]:
        sampled = []
        sample_id = start_id
        self.lstm.set_state(h)

        for _ in range(sample_size):
            x = np.array(sample_id).reshape((1, 1))
            out = self.embed.forward(x)
            out = self.lstm.forward(out)
            score = self.affine.forward(out)

            sample_id = np.argmax(score.flatten())
            sampled.append(int(sample_id))

        return sampled

class Seq2seq(BaseModel):
    # Seq2seq の初期化。
    # param: self: Seq2seq。
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

        self.encoder = Encoder(V, D, H)
        self.decoder = Decoder(V, D, H)
        self.softmax = TimeSoftmaxWithLoss()

        self.params = self.encoder.params + self.decoder.params
        self.grads = self.encoder.grads + self.decoder.grads
    
    # 順伝播。
    # param: self: Seq2seq。
    # param: xs: 入力データ。(batch_size, time_steps, hidden_size) の形状を持つ。
    # param: ts: 教師データ。(batch_size, time_steps, hidden_size) の形状を持つ。
    # return: loss: 損失値。
    def forward(
        self, 
        xs: np.ndarray[np.int32], 
        ts: np.ndarray[np.int32]
    ) -> np.float32:
        decoder_xs, decoder_ts = ts[:, :-1], ts[:, 1:]
        h = self.encoder.forward(xs)
        score = self.decoder.forward(decoder_xs, h)
        loss = self.softmax.forward(score, decoder_ts)
        return loss
    
    # 逆伝播。
    # param: self: Seq2seq。
    # param: dout: 逆伝播の勾配。
    # return: dout: 逆伝播の勾配。(batch_size, time_steps) の形状を持つ。
    def backward(
        self, 
        dout: np.float32 = 1
    ) -> np.ndarray[np.float32]:
        dout = self.softmax.backward(dout)
        dh = self.decoder.backward(dout)
        dout = self.encoder.backward(dh)
        return dout
    
    # 文章生成。
    # param: self: Seq2seq。
    # param: xs: 入力データ。(batch_size, time_steps, hidden_size) の形状を持つ。
    # param: start_id: 文章生成の開始単語ID。
    # param: sample_size: 文章生成の単語数。
    # return: sampled: 生成された文章の単語IDのリスト。
    def generate(
        self, 
        xs: np.ndarray[np.int32], 
        start_id: np.int32, 
        sample_size: np.int32
    ) -> np.ndarray[np.int32]:
        h = self.encoder.forward(xs)
        sampled = self.decoder.generate(h, start_id, sample_size)
        return sampled