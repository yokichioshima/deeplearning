import numpy as np
from layers.sigmoid_with_loss import SigmoidWithLoss
from layers.embedding_dot import EmbeddingDot
from samplers.unigram_sampler import UnigramSampler

class NegativeSamplingLoss:
    # NegativeSamplingLoss の初期化
    # param: self: NegativeSamplingLoss。
    # param: W: 出力層の重み。2次元配列で、各行が単語ベクトル。
    # param: corpus: コーパス(単語 ID リスト)。
    # param: power: 単語の出現頻度のべき乗。
    # param: sample_size: 負例のサンプリング数。
    def __init__(
        self, 
        W: np.ndarray[np.float32], 
        corpus: list[int],
        power: float =0.75, 
        sample_size: int =5
    ):
        self.sample_size = sample_size
        self.sampler = UnigramSampler(corpus, power, sample_size)
        self.loss_layers = [SigmoidWithLoss() for _ in range(sample_size + 1)]
        self.embed_dot_layers = [EmbeddingDot(W) for _ in range(sample_size + 1)]

        self.params, self.grads = [], []
        for layer in self.embed_dot_layers:
            self.params += layer.params
            self.grads += layer.grads
    
    # 順伝播
    # param: self: NegativeSamplingLoss。
    # param: h: 隠れ層の出力。
    # param: target: ターゲット。1 次元配列で、各行がターゲットの単語 ID。
    # return: loss: 損失値。
    def forward(
        self, 
        h: np.ndarray[np.float32], 
        target: np.ndarray[np.int32]
    ) -> float:
        batch_size = target.shape[0]
        negative_sample = self.sampler.get_negative_sample(target)

        # 正例の計算
        score = self.embed_dot_layers[0].forward(h, target)
        correct_label = np.ones(batch_size, dtype=np.int32)
        loss = self.loss_layers[0].forward(score, correct_label)

        # 負例の計算
        negative_label = np.zeros(batch_size, dtype=np.int32)
        for i in range(self.sample_size):
            negative_target = negative_sample[:, i]
            score = self.embed_dot_layers[i + 1].forward(h, negative_target)
            loss += self.loss_layers[i + 1].forward(score, negative_label)

        return loss
    
    # 逆伝播
    # param: self: NegativeSamplingLoss。
    # param: dout: 逆伝播の勾配。
    # return: dh: 隠れ層の勾配。
    def backward(self, dout=1):
        dh = 0
        for l0, l1 in zip(self.loss_layers, self.embed_dot_layers):
            dscore = l0.backward(dout)
            dh += l1.backward(dscore)[0]
        return dh