import sys
sys.path.append('..')
import numpy as np
from models.better_rnnlm_gen import BetterRnnlmGen
from dataset.ptb import load_data

corpus, word_to_id, id_to_word = load_data('train')
vocab_size = len(word_to_id)
model = BetterRnnlmGen()
model.load_params('../BetterRnnlm.pkl')

# start 文字と skip 文字の ID を取得する。
start_word = 'you'
start_id = word_to_id[start_word]
skip_words = ['N', '<unk>', '$']
skip_ids = [word_to_id[w] for w in skip_words]

# 文章生成
word_ids = model.generate(start_id, skip_ids)
txt = ' '.join([id_to_word[i] for i in word_ids])
txt = txt.replace(' <eos>', '.\n')
print(txt)

model.reset_state()

start_word = 'the meaning of life is'
start_ids = [word_to_id[w] for w in start_word.split(' ')]

for x in start_ids[:-1]:
    x = np.array(x).reshape(1, 1)
    model.predict(x)

word_ids = model.generate(start_ids[-1], skip_ids)
word_ids = start_ids[:-1] + word_ids
txt = ' '.join([id_to_word[i] for i in word_ids])
txt = txt.replace(' <eos>', '.\n')
print('-' * 50)
print(txt)