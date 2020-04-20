from keras.layers.embeddings import Embedding
from keras.preprocessing import sequence
from keras.models import load_model, Model
from sklearn.model_selection import train_test_split
from keras.layers import BatchNormalization
from keras.preprocessing.text import Tokenizer
from keras.layers import Dense, Input, Dropout, LSTM, Bidirectional, Activation, GlobalAveragePooling1D, Flatten
from keras.preprocessing.sequence import pad_sequences
from keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from keras.regularizers import L1L2
from keras import callbacks as ck
from keras.initializers import glorot_uniform
import numpy as np
import regex as re
import os
import pickle
import matplotlib.pyplot as plt


# embedding_dim = 100


class myCallback(ck.Callback):
  def on_epoch_end(self, epoch, logs={}):
    if(logs.get('accuracy')>0.6):
      print("\nReached 90% accuracy so cancelling training!")
      self.model.stop_training = True
callback = myCallback()


def text_read(file_name):
    data_list = []
    with open(file_name, 'r') as f:
        for line in f:
            line = line.strip()
            label = ' '.join(line[:line.find("]")].strip().split())
            text = line[line.find("]") + 1:].strip()
            data_list.append([label, text])

    return data_list


def labels_read(text_list):
    label_list = []
    text_list = [text_list[i][0].replace('[', '') for i in range(len(text_list))]
    label_list = [list(np.fromstring(text_list[i], dtype=float, sep=' ')) for i in range(len(text_list))]
    return label_list


def msg_read(text_list):
    msg_list = []
    msg_list = [text_list[i][1] for i in range(len(text_list))]
    return msg_list


def read_glove_vector():
    embeddings_index = {}
    words = set()
    coefs = {}
    with open('glove.6B.50d.txt', 'r', encoding='UTF-8') as f:
        for line in f:
            values = line.strip().split()
            values[0] = re.sub('[^a-zA-Z]', '', values[0])
            if len(values[0]) > 0:
                words.add(values[0])
                coefs[values[0]] = np.array(values[1:], dtype= np.float64)
        i = 1
        word_index = {}
        index_word = {}

        for word in sorted(words):
            word_index[word] = i
            index_word[i] = word
            i = i + 1
            vocab_size = len(word_index) + 1
            embed_dim = coefs['word'].shape[0]
        # print(vocab_size)
        # print(embed_dim)

        embeddings_matrix = np.zeros((vocab_size, embed_dim))
        for word, i in word_index.items():
            embeddings_matrix[i,:] = coefs[word]
        embedding_layer = Embedding(vocab_size, embed_dim)
        embedding_layer.build((None,))
        embedding_layer.set_weights([embeddings_matrix])
        print(embeddings_matrix.shape)
        return embedding_layer


def sentences_to_indices(text_arr, word_index, max_len):
    m = text_arr.shape[0]
    X_indices = np.zeros((m, max_len))
    for i in range(m):
        sentence_words = [w.lower() for w in text_arr[i].split()]

        j = 0
        for w in sentence_words:
            X_indices[i, j] = word_index[w]
            j += 1

    return X_indices


def plot_graphs(history, string):
  plt.plot(history.history[string])
  plt.plot(history.history['val_'+string])
  plt.xlabel("Epochs")
  plt.ylabel(string)
  plt.legend([string, 'val_'+string])
  plt.show()


def lstm_model(inputshape, embedding_layer):
    sentences_indices = Input(shape=inputshape, dtype=np.int32)
    embedding_layer = embedding_layer
    embeddings = embedding_layer(sentences_indices)
    X = Bidirectional(LSTM(256, return_sequences=True, recurrent_dropout=0.2, dropout=0.20, bias_regularizer=L1L2(0.01, 0.02)))(embeddings)
    X = Bidirectional(LSTM(256, return_sequences=True, recurrent_dropout=0.2, dropout=0.20))(X)
    X = BatchNormalization()(X)
    X = Dropout(0.4)(X)
    X = Flatten()(X)
    X = Dense(128,activation='relu')(X)
    X = Dropout(0.40)(X)
    X = Dense(7, activation='softmax')(X)
    model = Model(sentences_indices, X)
    return model

if __name__ == "__main__":
    textlist = text_read('data.txt')
    label_list = labels_read(textlist)
    msg_list = msg_read(textlist)
    x_train, x_test, y_train, y_test = train_test_split(msg_list, label_list, test_size=0.20, stratify=label_list,random_state=1234)
    t = Tokenizer( lower=True, filters='', oov_token="<OOV>")
    t.fit_on_texts(msg_list)
    embeddings_matrix = read_glove_vector()
    x_train_tokenized = t.texts_to_sequences(x_train)
    x_test_tokenized = t.texts_to_sequences(x_test)
    max_len = 50
    X_train = pad_sequences(x_train_tokenized, padding='post', maxlen=max_len)
    X_test = pad_sequences(x_test_tokenized, padding='post', maxlen=max_len)
    with open('tokenizer.pickle', 'wb') as tokenizer:
        pickle.dump(t, tokenizer, protocol=pickle.HIGHEST_PROTOCOL)
    embedding_layer = read_glove_vector()
    model = lstm_model((max_len,), embedding_layer)
    print(model.summary())
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    history = model.fit(X_train, np.array(y_train), batch_size=32, epochs=100, validation_data=[X_test, np.array(y_test)], callbacks= [callback])
    plot_graphs(history, 'accuracy')
    plot_graphs(history, 'loss')
    model.save('emoji_weights.h5')
    loss, acc = model.evaluate(X_test, np.array(y_test))
    test_sent = t.texts_to_sequences(['Feeling Happy today'])
    test_sent = pad_sequences(test_sent, maxlen=max_len)
    pred = model.predict(test_sent)
    print(np.argmax(pred))
