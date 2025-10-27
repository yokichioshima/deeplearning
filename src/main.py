import matplotlib.pyplot as plt
import numpy as np
from dataset import spiral
from common.trainer import Trainer
from common.util import preprocess, create_co_matrix, cos_similarity, ppmi
from optimizers.sgd import SGD
from two_layer_net import TwoLayerNet

def main():
    # max_epoch = 300
    # batch_size = 30
    # hidden_size = 10
    # learning_rate = 1.0

    # x, t = spiral.load_data()
    # model = TwoLayerNet(input_size=2, hidden_size=hidden_size, output_size=3)
    # optimizer = SGD(lr=learning_rate)

    # trainer = Trainer(model, optimizer)
    # trainer.fit(x, t, max_epoch, batch_size, eval_interval=10)
    # trainer.plot()
    text = 'You say goodbye and I say hello.'
    corpus, word_to_id, id_to_word = preprocess(text)
    vocab_size = len(word_to_id)
    C = create_co_matrix(corpus, vocab_size)
    W = ppmi(C)

    U, S, V = np.linalg.svd(W)

    for word, word_id in word_to_id.items():
        plt.annotate(word, (U[word_id, 0], U[word_id, 1]))
    
    plt.scatter(U[:, 0], U[:, 1], alpha=0.5)
    plt.savefig('figure.png')

    np.set_printoptions(precision=3)
    print('covarience matrix')
    print(C)
    print('-'*50)
    print('PPMI')
    print(W)



if __name__ == '__main__':
    main()