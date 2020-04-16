from flask import Flask
from flask import request
from flask import jsonify
from flask import render_template
import os
import sys
import numpy as np
import tensorflow as tf
import pickle
import pdb
from keras.models import Model
from keras.models import load_model
from keras.preprocessing.sequence import pad_sequences

# import flask
# from flask import Flask, render_template, request, jsonify
# import tensorflow as tf
# from keras.layers.embeddings import Embedding
# from keras.layers.embeddings import Embedding
# from keras.preprocessing import sequence
# from keras.models import load_model, Model
# from sklearn.model_selection import train_test_split
# from keras.layers import BatchNormalization
# from keras.preprocessing.text import Tokenizer
# from keras.layers import Dense, Input, Dropout, LSTM, Bidirectional, Activation
# from keras.preprocessing.sequence import pad_sequences
# from keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
# from keras.regularizers import L1L2
# from keras.initializers import glorot_uniform
# import numpy as np
# import regex as re
# import os
# import keras.backend.tensorflow_backend as tb
# import pickle
# import json

app = Flask(__name__)


def get_model():
    global model
    model = load_model('emoji_weights.h5')
    print('Model Successfully loaded')


graph = tf.compat.v1.get_default_graph()

tokenizer = pickle.load(open('emoji-predictor/tokenizer.pickle', 'rb'))


@app.route('/')
def home():
    return render_template('home.html')


@app.route('/predict', methods=['POST'])
def predict():
    # global graph
    global tokenizer
    # with graph.as_default():
    max_len = 50
    text = request.form['name']
    test_sent = tokenizer.texts_to_sequences([text])
    test_sent = pad_sequences(test_sent, maxlen=max_len)
    pred = model.predict(test_sent)
    prediction = np.argmax(pred)
    print(prediction)
    response = {
        'prediction': int(np.argmax(pred)), 'ContentType': 'application/json'
    }
    return jsonify(response)
    # return json.dumps({'prediction': int(np.argmax(pred))},200,{'ContentType':'application/json'})


@app.route('/update', methods=['POST'])
def update():
    global graph
    global tokenizer
    with graph.as_default():
        max_len = 50
        text = request.form['sentence']
        test_sent = tokenizer.texts_to_sequences([text])
        test_sent = pad_sequences(test_sent, maxlen=max_len)
        test_sent = np.vstack([test_sent] * 5)
        actual_output = request.form['dropdown_value']
        output_hash = {
        'Happy': np.array([1., 0., 0., 0., 0., 0., 0.]),
        'Fear': np.array([0., 1., 0., 0., 0., 0., 0.]),
        'Anger': np.array([0., 0., 1., 0., 0., 0., 0.]),
        'Sadness': np.array([0., 0., 0., 1., 0., 0., 0.]),
        'Disgust': np.array([0., 0., 0., 0., 1., 0., 0.]),
        'Shame': np.array([0., 0., 0., 0., 0., 1., 0.]),
        'Guilt': np.array([0., 0., 0., 0., 0., 0., 1.]),
        }
        actual_output = output_hash[actual_output].reshape((1, 7))
        actual_output = np.vstack([actual_output] * 5)
        model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
        model.fit(test_sent, actual_output, epochs=10, batch_size=32, shuffle=True)
        model.save('emoji_model.h5')
        get_model()
        response = {
        'update_text': 'Updated the values!! Should work in next few attempts..'
        }
    return jsonify(response)


if __name__ == "__main__":
    get_model()
    app.run('127.0.0.1', debug=False, threaded=False)
