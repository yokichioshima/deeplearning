import sys
sys.path.append('..')
import os
import numpy as np
import pickle

class BaseModel:
    # BaseModel を初期化します。
    # param: self: BaseModel。
    def __init__(self):
        self.params, self.grads = None, None
    
    # 順伝播。
    # param: self: BaseModel。
    # param: *args: 可変長引数。
    def forward(self, *args):
        raise NoImplementedError
    
    # 逆伝播。
    # param: self: BaseModel。
    # param: *args: 可変長引数。
    def backward(self, *args):
        raise NoImplementedError
    
    # パラメータをファイルに保存します。
    # self: BaseModel。
    # file_name: 保存するファイル名。
    def save_params(self, file_name=None):
        if file_name is None:
            file_name = self.__class__.__name__ + '.pkl'
        
        params = [p.astype(np.float16) for p in self.params]
        
        with open(file_name, 'wb') as f:
            pickle.dump(params, f)
    
    # パラメータをファイルから読み込みます。
    # param: self: BaseModel。
    # param: file_name: 読み込むファイル名。
    def load_params(self, file_name=None):
        if file_name is None:
            file_name = self.__class__.__name__ + '.pkl'
        
        if '/' in file_name:
            file_name = file_name.replace('/', os.sep)
        
        if not os.path.exists(file_name):
            raise IOError('No file:' + file_name)
        
        with open(file_name, 'rb') as f:
            params = pickle.load(f)
        
        params = [p.astype('f') for p in params]

        for i, param in enumerate(self.params):
            param[...] = params[i]