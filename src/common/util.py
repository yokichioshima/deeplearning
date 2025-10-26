import numpy as np


def preprocess(text):
    text = text.lower()
    text.replace('.', ' .')
    words = text.split(' ')

    word_to_id = {}
    id_to_word = {}

    for word in words:
        if word not in word_to_id:
            new_id = len(word_to_id)
            word_to_id[word] = new_id
            id_to_word[new_id] = word
    
    corpus = np.array([word_to_id[w] for w in words])

    return corpus, word_to_id, id_to_word

# cosine 類似度の算出
# :param x: ベクトル
# :param y: ベクトル
# :param eps: 0 割り防止のための微小値
# return:
def cos_similarity(x, y, eps=1e-8):
    nx = x / (np.sqrt(np.sum(x ** 2)) + eps)
    ny = y / (np.sqrt(np.sum(y ** 2)) + eps)
    return np.dot(nx, ny)

# 共起行列の作成
# param: corpus: コーパス(単語 ID リスト)
# param: vocab_size: 語彙数
# param: window_size: ウィンドウサイズ(ウィンドウサイズが 1 のとき単語の左右 1 単語がコンテキスト)
# return: 共起行列
def create_co_matrix(corpus, vocab_size, window_size=1):
    corpus_size = len(corpus)
    co_matrix = np.zeros(vocab_size, vocab_size, dtype=np.int32)

    for idx, word_id in enumerate(corpus):
        for i in range(1, window_size):
            left_idx = idx - i
            right_idx = idx + i

            if left_idx >= 0:
                left_word_id = corpus[left_idx]
                co_matrix[idx, left_word_id] += 1
            
            if right_idx < corpus_size:
                right_word_id = corpus[right_idx]
                co_matrix[idx, right_word_id] += 1
        
    return co_matrix