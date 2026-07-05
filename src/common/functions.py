import numpy as np

# softmax 関数。
# param: x: batch 処理の場合、 2 次元配列、そうでない場合は 1 次元配列。
# return: softmax 関数の出力。
def softmax(x: np.ndarray[np.float32]) -> np.ndarray[np.float32]:
    if x.ndim == 2:
        x = x - x.max(axis=1, keepdims=True)
        x = np.exp(x)
        x /= x.sum(axis=1, keepdims=True)
    elif x.ndim == 1:
        x = x - np.max(x)
        x = np.exp(x) / np.sum(np.exp(x))
       
    return x

# 交差エントロピー誤差の計算。
# param: y: 予測確率分布。batch 処理のを想定して2次元配列。
# param: t: 教師データ(正解ラベル)。
# return: 交差エントロピー誤差。
def cross_entropy_error(
    y: np.ndarray[np.float32], 
    t: np.ndarray[np.int32]
) -> np.float32:
    if y.ndim == 1:
        t = t.reshape(1, t.size)
        y = y.reshape(1, y.size)
    
    # 教師データが one-hot ベクトルの場合、正解ラベルのインデックスに変換
    if t.size == y.size:
        t = t.argmax(axis=1)
    
    batch_size = y.shape[0]
    return -np.sum(np.log(y[np.arange(batch_size), t] + 1e-7)) / batch_size

# softmax 関数と交差エントロピー誤差の計算をまとめた関数。
# param: X: 入力データ。batch 処理のを想定して2次元配列。
# param: t: 教師データ(正解ラベル)。
# return: 交差エントロピー誤差。
def softmax_loss(
    X: np.ndarray[np.float32], 
    t: np.ndarray[np.int32]
) -> np.float32:
    y = softmax(X)
    return cross_entropy_error(y, t)