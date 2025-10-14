import numpy as np;
from two_layer_net import TwoLayerNet;

def main():
    model = TwoLayerNet(2, 4, 2)
    x = np.random.rand(2)
    y = model.predict(x)
    print(y)


if __name__ == '__main__':
    main()