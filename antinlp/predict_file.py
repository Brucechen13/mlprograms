# -*- coding:UTF-8
import sys
reload(sys)
sys.setdefaultencoding('UTF-8')
import keras
import numpy as np
import pandas as pd
from collections import defaultdict
from gensim.corpora import Dictionary
from gensim.models import LdaMulticore
from sklearn.metrics.pairwise import cosine_distances, euclidean_distances
import numpy as np

# from tflearn.data_utils import pad_sequences
import json
import jieba

from langconv import *

import tensorflow as tf
import keras.backend.tensorflow_backend as KTF
from sklearn.externals import joblib
import xgboost as xgb

config = tf.ConfigProto()
config.gpu_options.allow_growth=True   #不全部占满显存, 按需分配
sess = tf.Session(config=config)

KTF.set_session(sess)

EMBEDDING_DIM = 300
VOCAB_LENGTH = 3000

from tensorflow.python.client import device_lib

print(device_lib.list_local_devices())

from features_all import ngram_features, lda_features, occur_features, simple_static_features, tfidf_features
from models import AttLayer

jieba.add_word('收钱码')
jieba.add_word('借呗码')
jieba.add_word('花呗')
jieba.add_word('借呗')
RANDOM_SEED = 42

PAD_ID = 0

def cht_to_chs(line):
    line = Converter('zh-hans').convert(line)
    line.encode('utf-8')
    return line
def parse_df(df):
    size = df.shape[0]
    parse_data = df.copy()
    for i in range(size):
        parse_data.iloc[i, 1] = cht_to_chs(df.iloc[i, 1].decode('UTF-8'))
        parse_data.iloc[i, 2] = cht_to_chs(df.iloc[i, 2].decode('UTF-8'))
    return parse_data

def pad_sequences(x_list_,max_sentence_len):
    length_x = len(x_list_)
    x_list=[]
    for i in range(0, max_sentence_len):
        if i < length_x:
            x_list.append(x_list_[i])
        else:
            x_list.append(PAD_ID)
    return x_list

def translate(text, translation):
    for token, replacement in translation.items():
        text = text.replace(token, ' ' + replacement + ' ')
    text = text.replace('  ', ' ')
    return text
def token_string_as_list(string, token_string='char'):
    string = string.decode('UTF-8')
    translation = {
        '***':'*',
        '花被':'花呗'
    }
    translate(string, translation)
    length = len(string)
    if token_string == 'char':
        listt = [string[i] for i in range(length)]
    elif token_string == 'word':
        listt = jieba.lcut(string)
    listt = [item for item in listt if item.strip()]
    return listt


def parse_train_data(train_data, vocab_indexdict, token_string):
    size = train_data.shape[0]
    parse_data = train_data.copy()
    #import pdb;pdb.set_trace()
    for i in range(size):
        token_list1 = token_string_as_list(train_data.iloc[i, 1], token_string=token_string)
        token_list2 = token_string_as_list(train_data.iloc[i, 2], token_string=token_string)
        parse_data.iloc[i, 1] = ' '.join([str(vocab_indexdict[item]) for item in token_list1 if item in vocab_indexdict])
        parse_data.iloc[i, 2] = ' '.join([str(vocab_indexdict[item]) for item in token_list2 if item in vocab_indexdict])
    return parse_data

def predict(model, X_q1, X_q2, X_magic):
    y1 = model.predict([X_q1, X_q2, X_magic], batch_size=1024, verbose=1).reshape(-1)
    y2 = model.predict([X_q2, X_q1, X_magic], batch_size=1024, verbose=1).reshape(-1)
    return (y1 + y2) / 2

def parse_model(model_path, train_data, feas, MAX_SEQUENCE_LENGTH):
    word_squence_ques1 = list(train_data.iloc[:, 1])
    word_squence_ques1 = [[int(im) for im in item.split(' ')] for item in word_squence_ques1]
    word_squence_ques2 = list(train_data.iloc[:, 2])
    word_squence_ques2 = [[int(im) for im in item.split(' ')] for item in word_squence_ques2]
    # MAX_SEQUENCE_LENGTH = MAX_SEQUENCE_LENGTH # char 40 word 26
    x1 = np.array([pad_sequences(item, MAX_SEQUENCE_LENGTH) for item in word_squence_ques1])
    x2 = np.array([pad_sequences(item, MAX_SEQUENCE_LENGTH) for item in word_squence_ques2])

    X_train_q1 = x1
    X_train_q2 = x2
    X_train_magic = feas
    #import pdb;pdb.set_trace()
    model = keras.models.load_model(model_path, custom_objects={'AttLayer':AttLayer})
    pred = (np.asarray(model.predict([X_train_q1,X_train_q2,
                            X_train_magic])))#
    return pred

def parse_model_all(model_path, train_data_char, train_data_word, feas, MAX_SEQUENCE_LENGTH_CHAR, MAX_SEQUENCE_LENGTH_WORD):
    word_squence_ques1 = list(train_data_char.iloc[:, 1])
    word_squence_ques1 = [[int(im) for im in item.split(' ')] for item in word_squence_ques1]
    word_squence_ques2 = list(train_data_char.iloc[:, 2])
    word_squence_ques2 = [[int(im) for im in item.split(' ')] for item in word_squence_ques2]
    # MAX_SEQUENCE_LENGTH = MAX_SEQUENCE_LENGTH # char 40 word 26
    x1_char = np.array([pad_sequences(item, MAX_SEQUENCE_LENGTH_CHAR) for item in word_squence_ques1])
    x2_char = np.array([pad_sequences(item, MAX_SEQUENCE_LENGTH_CHAR) for item in word_squence_ques2])

    word_squence_ques1 = list(train_data_word.iloc[:, 1])
    word_squence_ques1 = [[int(im) for im in item.split(' ')] for item in word_squence_ques1]
    word_squence_ques2 = list(train_data_word.iloc[:, 2])
    word_squence_ques2 = [[int(im) for im in item.split(' ')] for item in word_squence_ques2]
    # MAX_SEQUENCE_LENGTH = MAX_SEQUENCE_LENGTH # char 40 word 26
    x1_word = np.array([pad_sequences(item, MAX_SEQUENCE_LENGTH_WORD) for item in word_squence_ques1])
    x2_word = np.array([pad_sequences(item, MAX_SEQUENCE_LENGTH_WORD) for item in word_squence_ques2])

    X_train_q1 = x1_char
    X_train_q2 = x2_char
    X_train_magic = feas
    X_train_q10 = x1_word
    X_train_q20 = x2_word
    #import pdb;pdb.set_trace()
    model = keras.models.load_model(model_path, custom_objects={'AttLayer':AttLayer})
    pred = (np.asarray(model.predict([X_train_q1,X_train_q2, X_train_q10, X_train_q20
                            ,X_train_magic])))#
    return pred

def parse_features(train_data):
    lda_feas = lda_features(train_data)
    ngram_feas = ngram_features(train_data)
    simsummary_feas = simple_static_features(train_data)
    all_feas = np.concatenate([ngram_feas, lda_feas, simsummary_feas], axis=1)
    print np.isnan(all_feas).any()
    return all_feas

paths = ['data/checkpoints/stacking-word-char-rnn-checkpoint.h5',
                 'data/checkpoints/stacking-word-char-cnn-checkpoint.h5',
                 'data/checkpoints/stacking-word-char-mix-checkpoint.h5',
                'data/checkpoints/stacking-word-word-rnn-checkpoint.h5',
                 'data/checkpoints/stacking-word-word-cnn-checkpoint.h5',
                 'data/checkpoints/stacking-word-word-mix-checkpoint.h5']
model_path = 'data/checkpoints/test/4weights.22.hdf5'

def predict_output(filein, fileout):
    # bst_new = xgb.Booster({'nthread':4}) #init model
    # bst_new.load_model("xgb.model") # load data
    train_data_ori = pd.read_csv(filein, sep='\t', header=None)
    train_data_ori = parse_df(train_data_ori)
    with open('data/aux/vocab_indexdict_word.json') as f:
        vocab_indexdict_word = json.load(f)
    with open('data/aux/vocab_indexdict_char.json') as f:
        vocab_indexdict_char = json.load(f)
    train_data_char = parse_train_data(train_data_ori, vocab_indexdict_char,token_string='char')
    train_data_word = parse_train_data(train_data_ori, vocab_indexdict_word,token_string='word')

    all_features_char = parse_features(train_data_char)
    all_features_word = parse_features(train_data_word)
    tfidf_feas = tfidf_features(train_data_ori)
    all_features =np.concatenate([all_features_char, tfidf_feas], axis=1)
    preds = []
    #pred = parse_model(model_path, train_data_char, all_features, 40)
    pred = parse_model_all(model_path, train_data_char, train_data_word, all_features, 40, 30)
    #X_train_stack = np.concatenate(preds, axis=1)
    #dtest = xgb.DMatrix(X_train_stack)
    #pred1 = bst_new.predict(dtest)
    train_data = train_data_ori
    y_pred = save_result_by_logit(pred, list(train_data.iloc[:,0]), fileout)
    if train_data.shape[1] > 3:
        from sklearn.metrics import f1_score
        _val_f1 = f1_score(list(train_data.iloc[:, 3]), y_pred)
        print ('f1-score', _val_f1)
        f_obj = open('data/test/log_err.txt', mode='w')
        for i in range(train_data.shape[0]):
            if train_data.iloc[i, 3] != y_pred[i]:
                f_obj.write(str(train_data.iloc[i, 0]) + '\t' + str(train_data.iloc[i, 1]) + '\t' + str(train_data.iloc[i, 2]) + '\t')
                f_obj.write(str(train_data.iloc[i, 3]) + '\t' + str(y_pred[i]) + '\n')
        f_obj.close()

def save_result_by_label(label, line_no_list, outpath):
    file_object = open(outpath, 'a')
    val_pred = []
    for l, lineid in zip(label, line_no_list):
        file_object.write(str(lineid) + "\t" + str(l) + "\n")
    file_object.close()
    return label


def save_result_by_logit(logits, line_no_list, outpath):
    label = [1 if item > 0.5 else 0 for item in logits]
    return save_result_by_label(label, line_no_list, outpath)


if __name__ == '__main__':
    predict_output(sys.argv[1], sys.argv[2])
