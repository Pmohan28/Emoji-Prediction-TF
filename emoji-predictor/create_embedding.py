from keras.layers.embeddings import Embedding
from keras.preprocessing import sequence
from keras.models import load_model, Model
from sklearn.model_selection import train_test_split
from keras.layers import BatchNormalization
from keras.preprocessing.text import Tokenizer
from keras.layers import Dense, Input, Dropout, LSTM, Bidirectional, Activation
from keras.preprocessing.sequence import pad_sequences
from keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from keras.regularizers import L1L2
from keras.initializers import glorot_uniform
import numpy as np
import regex as re
import os
import pickle


def read_text_file(file_name):
    data_list = []
    with open(file_name, 'r') as f:
        for line in f:
            line = line.strip()
            label = ' '.join(line[:line.find("]")].strip().split())
            text = line[line.find("]") + 1:].strip()
            data_list.append([label, text])

    return data_list


def extract_labels(text_list):
    label_list = []
    text_list = [text_list[i][0].replace('[', '') for i in range(len(text_list))]
    label_list = [list(np.fromstring(text_list[i], dtype=float, sep=' ')) for i in range(len(text_list))]
    return label_list


def extract_text_msgs(text_list):
    msg_list = []
    msg_list = [text_list[i][1] for i in range(len(text_list))]
    return msg_list


def read_glove_vector(glovefile):
    with open(glovefile, 'r', encoding='UTF-8') as f:
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

    return word_to_index, index_to_word, word_to_vec


def sentences_to_indices(text_arr, word_to_index, max_len):
    m = text_arr.shape[0]
    X_indices = np.zeros((m, max_len))
    for i in range(m):
        sentence_words = [w.lower() for w in text_arr[i].split()]

        j = 0
        for w in sentence_words:
            X_indices[i, j] = word_to_index[w]
            j += 1

    return X_indices


def create_embeddings_layer(word_to_index, word_to_vec):
    corpus_len = len(word_to_index) + 1
    embedding_dim = word_to_vec['word'].shape[0]
    embed_matrix = np.zeros(shape=(corpus_len, embedding_dim))

    for word, index in word_to_index.items():
        embed_matrix[index, :] = word_to_vec[word]

    embedding_layer = Embedding(corpus_len, embedding_dim)

    embedding_layer.build((None,))
    embedding_layer.set_weights([embed_matrix])
    # print(embedding_layer.get_weights()[0][1][3])
    return embedding_layer


def create_lstm_model(inputshape, embedding_layer):
    sentences_indices = Input(shape=inputshape, dtype=np.int32)
    embedding_layer = embedding_layer
    embeddings = embedding_layer(sentences_indices)
    X = Bidirectional(LSTM(256, return_sequences=True, recurrent_dropout=0.2, dropout=0.20, bias_regularizer=L1L2(0.01, 0.02)))(embeddings)
    X = BatchNormalization()(X)
    X = Dropout(0.4)(X)
    X = LSTM(128)(X)
    X= Dropout(0.40)(X)
    X = Dense(7, activation='softmax')(X)
    model = Model(sentences_indices, X)
    return model




if __name__ == "__main__":
    textlist = read_text_file('data.txt')
    label_list = extract_labels(textlist)
    msg_list = extract_text_msgs(textlist)
    word_to_index, index_to_word, word_to_vec = read_glove_vector('glove.6B.50d.txt')
    x_train, x_test, y_train, y_test = train_test_split(msg_list, label_list, test_size=0.20, stratify=label_list,
                                                        random_state=1234)
    t = Tokenizer(lower=True, filters='')
    t.fit_on_texts(msg_list)
    x_train_tokenized = t.texts_to_sequences(x_train)
    x_test_tokenized = t.texts_to_sequences(x_test)
    max_len = 50
    X_train = pad_sequences(x_train_tokenized, padding='post', maxlen=max_len)
    X_test = pad_sequences(x_test_tokenized, padding='post', maxlen=max_len)
    # if os.path.exists('tokenizer.pickle'):
    #     os.remove('tokenizer.pickle')
    with open('tokenizer.pickle', 'wb') as tokenizer:
        pickle.dump(t, tokenizer, protocol=pickle.HIGHEST_PROTOCOL)
    embedding_layer = create_embeddings_layer(word_to_index, word_to_vec)
    model = create_lstm_model((max_len,), embedding_layer)
    print(model.summary())
    print(np.array(y_train).shape)
    # callbacks = [
    #     EarlyStopping(patience=12, verbose=1),
    #     ReduceLROnPlateau(factor=0.1, patience=5, min_lr=0.000001, verbose=1),
    #     ModelCheckpoint('./emoji-{epoch:03d}.h5', verbose=1, save_weights_only=True)
    # ]
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    model.fit(X_train, np.array(y_train), batch_size=32, epochs=32, validation_data=[X_test, np.array(y_test)])
    model.save('emoji_weights.h5')
    loss, acc = model.evaluate(X_test, np.array(y_test))
    test_sent = t.texts_to_sequences(['Feeling Happy about the day'])
    test_sent = pad_sequences(test_sent, maxlen=max_len)
    pred = model.predict(test_sent)
    print(np.argmax(pred))



