import numpy as np

# 前処理
# param: text: テキストデータ(文字列)
# return: corpus: 単語 ID リスト
#         word_to_id: 単語から単語 ID へのディクショナリ
#         id_to_word: 単語 ID から単語へのディクショナリ
def preprocess(text: str) -> tuple[
    np.ndarray[np.int32], 
    dict[str, int], 
    dict[int, str]
]:
    text = text.lower()
    text = text.replace('.', ' .')
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
def cos_similarity(
    x: np.ndarray[np.float32],
    y: np.ndarray[np.float32], 
    eps=1e-8
) -> float:
    nx = x / (np.sqrt(np.sum(x ** 2)) + eps)
    ny = y / (np.sqrt(np.sum(y ** 2)) + eps)
    return np.dot(nx, ny)


# 類似単語の検索
# param: query: クエリ(テキスト)
# param: word_to_id: 単語から単語 ID へのディクショナリ
# param: id_to_word: 単語 ID から単語へのディクショナリ
# param: word_matrix: 単語ベクトルをまとめた行列。各行に対応する単語のベクトルが格納されていることを想定
# param: top: 上位何位魔で表示するか
def most_similar(
    query: str, 
    word_to_id: dict[str, int], 
    id_to_word: dict[int, str], 
    word_matrix: np.ndarray[np.float32], 
    top: int = 5
):
    if query not in word_to_id:
        print('%s is not found' % query)
        return
    
    print('\n[query] ' + query)
    query_id = word_to_id[query]
    query_vec = word_matrix[query_id]

    vocab_size = len(id_to_word)
    similarity = np.zeros(vocab_size)
    for i in range(vocab_size):
        similarity[i] = cos_similarity(word_matrix[i], query_vec)
    
    count = 0
    for i in (-1 * similarity).argsort():
        if id_to_word[i] == query:
            continue
        print(' %s: %s' % (id_to_word[i], similarity[i]))

        count += 1
        if count >= top:
            return

# one-hot 表現への変換
# param: corpus: 単語 ID リスト(1 次元もしくは 2 次元の Numpy 配列)
# param: vocab_size: 語彙数
# return: one-hot 表現(2 次元もしくは 3 次元の Numpy 配列)
def convert_one_hot(corpus: np.ndarray[np.int32], vocab_size: int) -> np.ndarray[np.int32]:
    N = corpus.shape[0]

    if corpus.ndim == 1:
        one_hot = np.zeros((N, vocab_size), dtype=np.int32)
        for idx, word_id in enumerate(corpus):
            one_hot[idx, word_id] = 1
        return one_hot

        
    elif corpus.ndim == 2:
        C = corpus.shape[1]
        one_hot = np.zeros((N, C, vocab_size), dtype=np.int32)
        for idx_0, word_ids in enumerate(corpus):
            for idx_1, word_id in enumerate(word_ids):
                one_hot[idx_0, idx_1, word_id] = 1
        return one_hot

    else:
        raise ValueError('corpus.ndim must be 1 or 2')


# 共起行列の作成
# param: corpus: コーパス(単語 ID リスト)
# param: vocab_size: 語彙数
# param: window_size: ウィンドウサイズ(ウィンドウサイズが 1 のとき単語の左右 1 単語がコンテキスト)
# return: 共起行列
def create_co_matrix(
    corpus: np.ndarray[np.int32], 
    vocab_size: int, 
    window_size: int = 1
) -> np.ndarray[np.int32]:
    corpus_size = len(corpus)
    co_matrix = np.zeros((vocab_size, vocab_size), dtype=np.int32)

    for idx, word_id in enumerate(corpus):
        for i in range(1, window_size + 1):
            left_idx = idx - i
            right_idx = idx + i

            if left_idx >= 0:
                left_word_id = corpus[left_idx]
                co_matrix[word_id, left_word_id] += 1
            
            if right_idx < corpus_size:
                right_word_id = corpus[right_idx]
                co_matrix[word_id, right_word_id] += 1
        
    return co_matrix

# PPMI (正の相互情報量)の作成
# param C: 共起行列
# param verbose: 進行状況を出力するかどうか
# return: PPMI 行列
def ppmi(C: np.ndarray[np.int32], verbose=False, eps= 1e-8) -> np.ndarray[np.float32]:
    M = np.zeros_like(C, dtype=np.float32)
    N = np.sum(C)
    S = np.sum(C, axis=0)
    total = C.shape[0] * C.shape[1]
    cnt = 0

    for i in range(C.shape[0]):
        for j in range(C.shape[1]):
            pmi = np.log2(C[i, j] * N / (S[j]*S[i]) + eps)
            M[i, j] = max(0, pmi)

            if verbose:
                cnt += 1
                if cnt % (total//100 + 1) == 0:
                    print('%.1f%% done' % (100*cnt/total))
    return M

# コンテキストとターゲットの作成
# param: corpus: コーパス(単語 ID リスト)
# param: window_size: ウィンドウサイズ(ウィンドウサイズが 1 のとき単語の左右 1 単語がコンテキスト)
# return: contexts: コンテキスト、
#         target: ターゲット
def create_contexts_target(
    corpus: np.ndarray[np.int32], 
    window_size: int = 1
) -> tuple[
    np.ndarray[np.int32], 
    np.ndarray[np.int32]
]:
    target = corpus[window_size:-window_size]
    contexts = []

    for idx in range(window_size, len(corpus) - window_size):
        cs = []
        for t in range(-window_size, window_size + 1):
            if t == 0:
                continue
            cs.append(corpus[idx + t])
        contexts.append(cs)
    
    return np.array(contexts), np.array(target)