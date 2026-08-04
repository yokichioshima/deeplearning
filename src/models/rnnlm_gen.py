import sys
sys.path.append('..')
import numpy as np
from common.functions import softmax
from models.rnnlm import Rnnlm
from models.better_rnnlm import BetterRnnlm

class RnnlmGen(Rnnlm):
    # 文章生成
    # param: self: RnnlmGen layer。
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
            score = self.predict(x)
            p = softmax(score.flatten())

            sampled = np.random.choice(len(p), size=1, p=p)
            if (skip_ids is None) or (sampled not in skip_ids):
                word_ids.append(int(sampled))
            x = sampled

        return word_ids
    
    # RNN の状態を取得する。
    # param: self: RnnlmGen layer。
    # return: RNN の状態。
    def get_state(self) -> tuple[np.ndarray[np.float32], np.ndarray[np.float32]]:
        return self.lstm_layer.h, self.lstm_layer.c
    
    # RNN の状態を設定する。
    # param: self: RnnlmGen layer。
    # param: state: RNN の状態。
    # return: None。
    def set_state(
        self, 
        state: tuple[np.ndarray[np.float32], np.ndarray[np.float32]]
    ) -> None:
        self.lstm_layer.set_state(*state)