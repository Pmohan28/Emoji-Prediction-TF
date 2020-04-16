import regex as re
import numpy as np

def read_glove_vector(glovefile):
    with open('/content/gdrive/My Drive/NLP/Sequence Modelling/Weights/glove.6B.50d.txt', 'r', encoding='UTF-8') as f:
        words = set()
        word_to_vec = {}
        for line in f:
            line = line.strip().split()
            line[0] = re.sub('[^a-zA-Z]', '', line[0])
            if len(line[0]) > 0:
                words.add(line[0])
                word_to_vec[line[0]] = np.array(line[1:], dtype=np.float64)
                # print (word_to_vec[line[0]])
        i = 1
        word_to_index = {}
        index_to_word = {}

        for word in sorted(words):
            word_to_index[word] = i
            index_to_word[i] = word

            i = i + 1

    return word_to_vec, word_to_index, index_to_word
