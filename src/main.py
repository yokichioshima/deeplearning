from dataset import spiral;
import matplotlib.pyplot as plt;
import numpy as np;
from two_layer_net import TwoLayerNet;

def main():
    x, t = spiral.load_data()
    print('x', x.shape)
    print('y', t.shape)

    N = 100
    CLS_NUM = 3
    markers = ['o', 'x', '^']
    for i in range(CLS_NUM):
        plt.scatter(x[i*N:(i+1)*N, 0], x[i*N:(i+1)*N, 1], s=40, marker=markers[i])
    plt.savefig('figure.png')


if __name__ == '__main__':
    main()