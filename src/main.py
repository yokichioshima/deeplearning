from dataset import ptb

def main():
    corpus, wort_to_id, id_to_word = ptb.load_data('train')

    print('corpus size:', len(corpus))
    print('corpus[:30]', corpus[:30])
    print()
    print('id_to_word[0]:', id_to_word[0])
    print('id_to_word[1]:', id_to_word[1])
    print('id_to_word[2]:', id_to_word[2])
    print()
    print("word_to_id['car']:", wort_to_id['car'])
    print("word_to_id['happy']:", wort_to_id['happy'])
    print("word_to_id['lexus']:", wort_to_id['lexus'])




if __name__ == '__main__':
    main()