import numpy as np


def read_text_file(filename):
    data_list = []
    with open(filename, 'r') as f:
        for line in f:
            line = line.strip()
            label = ''.join(line[:line.find(']')].strip().split())
            text = line[line.find(']')+1:].strip()
            data_list.append([label, text])

    return data_list


def extract_labels(text_list):
    label_list = []
    text_list = [text_list[i][0].replace('[', '') for i in range(len(text_list))]
    label_list = [list(np.fromstring(text_list[i], dtype= float, sep = '')) for i in range(len(text_list))]
    return label_list


def extract_test_msgs(text_list):
    msg_list = []
    msg_list = [text_list[i][1] for i in range (len(text_list))]
    return msg_list



