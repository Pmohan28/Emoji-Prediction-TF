import requests
import pickle
import os
import numpy as np

def main():
    file = 'raw.pickle'
    response = requests.get('https://raw.githubusercontent.com/bfelbo/DeepMoji/master/data/PsychExp/raw.pickle')
    open(file, 'wb').write(response.content)
    data = pickle.load(open(file, 'rb'), encoding='latin1')
    if os.path.exists('data.txt'):
        os.remove('data.txt')
    try:
        texts = [str(x) for x in data['texts']]
        labels = [x['label']for x in data['info']]
        with open('data.txt', 'a') as txtfile:
            for i in range(len(texts)):
                txtfile.write(np.array2string(labels[i]))
                txtfile.write((str(texts[i]) + '\n'))

    except Exception as e:
        print('An exception occurred')





if __name__ == "__main__":
    main()
