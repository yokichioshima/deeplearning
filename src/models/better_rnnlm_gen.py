import sys
sys.path.append('..')
import numpy as np
from common.functions import softmax
from models.better_rnnlm import BetterRnnlm

class BetterRnnlmGen(BetterRnnlm):
    # 文章生成
    # param: self: BetterRnnlmGen layer。
    # param: start_id: 文章生成の開始単語の ID 。
    # param: skip_ids: 文章生成の際にスキップする単語の ID のリスト。
    # param: sample_size: 文章生成の際に生成する単語数。
    def generate(
        self, 
        start_id: np.int32, 
        skip_ids: np.ndarray[np.int32] =None, 
        sample_size: np.int32 =100
    ) -> np.ndarray[np.int32]:
        word_ids = [start_id]

        x = start_id
        while len(word_ids) < sample_size:
            x = np.array(x).reshape(1, 1)
            score = self.predict(x).flatten()
            p = softmax(score).flatten()

            sampled = np.random.choice(len(p), size=1, p=p)
            if (skip_ids is None) or (sampled not in skip_ids):
                x = sampled
                word_ids.append(int(sampled))

        return word_ids
    
    # RNN の状態を取得する。
    # param: self: BetterRnnlmGen layer。
    # return: RNN の状態。
    def get_state(self) -> list[tuple[np.ndarray[np.float32], np.ndarray[np.float32]]]:
        states = []
        for layer in self.lstm_layers:
            states.append((layer.h, layer.c))
        return states
    
    # RNN の状態を設定する。
    # param: self: BetterRnnlmGen layer。
    # param: states: RNN の状態。
    # return: None。
    def set_state(
        self, 
        states: list[tuple[np.ndarray[np.float32], np.ndarray[np.float32]]]
    ) -> None:
        for layer, state in zip(self.lstm_layers, states):
            layer.set_state(*state)