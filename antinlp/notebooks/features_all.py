import pandas as pd
import numpy as np
from collections import defaultdict
from gensim.corpora import Dictionary
from gensim.models import LdaMulticore
from sklearn.metrics.pairwise import cosine_distances, euclidean_distances
import json

RANDOM_SEED = 42

NUM_TOPICS = 300

def ngram_features(test_data):
    def split_string_as_list_by_ngram(input_string,ngram_value):
        input_string="".join([string for string in input_string if string.strip()])
        length = len(input_string)
        result_string=[]
        for i in range(length):
            if i + ngram_value < length + 1:
                result_string.append(input_string[i:i+ngram_value])
        return result_string
    def compute_blue_ngram(x1_list,x2_list):
        """
        compute blue score use ngram information. x1_list as predict sentence,x2_list as target sentence
        :param x1_list:
        :param x2_list:
        :return:
        """
        count_dict={}
        count_dict_clip={}
        #1. count for each token at predict sentence side.
        for token in x1_list:
            if token not in count_dict:
                count_dict[token]=1
            else:
                count_dict[token]=count_dict[token]+1
        count=np.sum([value for key,value in count_dict.items()])
        #2.count for tokens existing in predict sentence for target sentence side.
        for token in x2_list:
            if token in count_dict:
                if token not in count_dict_clip:
                    count_dict_clip[token]=1
                else:
                    count_dict_clip[token]=count_dict_clip[token]+1
        #3. clip value to ceiling value for that token
        count_dict_clip={key:(value if value<=count_dict[key] else count_dict[key]) for key,value in count_dict_clip.items()}
        count_clip=np.sum([value for key,value in count_dict_clip.items()])
        result=float(count_clip)/(float(count)+0.00000001)
        return result
    def cal_ngram(csv_data, ngram_value):
        ngram_lt1 = []
        ngram_lt2 = []
        for i in range(csv_data.shape[0]):
            x1_list = csv_data.iloc[i, 1].split(' ')
            x2_list = csv_data.iloc[i, 2].split(' ')
            res1 = compute_blue_ngram(split_string_as_list_by_ngram(x1_list, ngram_value),
                                      split_string_as_list_by_ngram(x2_list,ngram_value))
            res2 = compute_blue_ngram(split_string_as_list_by_ngram(x2_list, ngram_value),
                                      split_string_as_list_by_ngram(x1_list,ngram_value))
            ngram_lt1.append(res1)
            ngram_lt2.append(res2)
        return ngram_lt1,ngram_lt2
    fea_dict = {}
    for ngram in range(3, 9):
        ngram_lt1,ngram_lt2 = cal_ngram(test_data, ngram)
        fea_dict['ngram1'+str(ngram)] = ngram_lt1
        fea_dict['ngram2'+str(ngram)] = ngram_lt2
    save_data = pd.DataFrame(fea_dict)
    return save_data.values

def lda_features(test_data):
    train_data = pd.read_csv('data/aux/train_char_indexvec.csv')
    documents = list(train_data.iloc[:, 1])
    documents.extend(list(train_data.iloc[:, 2]))
    documents = [item.split(' ') for item in documents]
    dictionary = Dictionary(documents)
    corpus = [dictionary.doc2bow(document) for document in documents]
    model = LdaMulticore(
        corpus,
        num_topics=NUM_TOPICS,
        id2word=dictionary,
        random_state=RANDOM_SEED,
    )
    def compute_topic_distances(pair):
        q1_bow = dictionary.doc2bow(pair[0])
        q2_bow = dictionary.doc2bow(pair[1])
        q1_topic_vec = np.array(model.get_document_topics(q1_bow, minimum_probability=0))[:, 1].reshape(1, -1)
        q2_topic_vec = np.array(model.get_document_topics(q2_bow, minimum_probability=0))[:, 1].reshape(1, -1)
        return [
            cosine_distances(q1_topic_vec, q2_topic_vec)[0][0],
            euclidean_distances(q1_topic_vec, q2_topic_vec)[0][0],
        ]
    cosine_lt = []
    euclidean_lt = []
    for i in range(test_data.shape[0]):
        cosine_val, euclidean_val = compute_topic_distances((test_data.iloc[i, 1].split(' '), test_data.iloc[i, 2].split(' ')))
        cosine_lt.append(cosine_val)
        euclidean_lt.append(euclidean_val)
    lda_feas = pd.DataFrame({'cosine_distances':cosine_lt, 'euclidean_distances':euclidean_lt})
    lda_feas = lda_feas.values
    return lda_feas

def occur_features(test_data):
    df_all_pairs = test_data.copy()
    columns = list(df_all_pairs.columns.values)
    columns[:3] = ['id', 'question1', 'question2']
    df_all_pairs.columns = columns
    df_unique_texts = pd.read_csv('data/test/occur_uniq_texts.csv')
    with open('data/test/occur_counts.json') as f:
        q_counts = json.load(f)
    question_ids = pd.Series(df_unique_texts.index.values, index=df_unique_texts['question'].values).to_dict()
    df_all_pairs['q1_id'] = df_all_pairs['question1'].map(question_ids)
    df_all_pairs['q2_id'] = df_all_pairs['question2'].map(question_ids)
    df_all_pairs['q1_freq'] = df_all_pairs['q1_id'].map(lambda x: q_counts.get(x, 0))
    df_all_pairs['q2_freq'] = df_all_pairs['q2_id'].map(lambda x: q_counts.get(x, 0))
    df_all_pairs['freq_ratio'] = df_all_pairs['q1_freq'] / df_all_pairs['q2_freq']
    df_all_pairs['freq_ratio_inverse'] = df_all_pairs['q2_freq'] / df_all_pairs['q1_freq']
    occur_feas = df_all_pairs.loc[:, ['q1_freq', 'q2_freq', 'freq_ratio', 'freq_ratio_inverse']]
    return occur_feas.values


def simple_static_features(test_data):
    def word_difference_ratio(q1_tokens, q2_tokens):
        return 1.0 * len(set(q1_tokens) ^ set(q2_tokens)) / (len(set(q1_tokens)) + len(set(q2_tokens)))

    def extract_tokenized_features(pair):
        q1 = pair[0]
        q2 = pair[1]
        shorter_token_length = min(len(q1), len(q2))
        longer_token_length = max(len(q1), len(q2))
        return [
            np.log(shorter_token_length + 1),
            np.log(longer_token_length + 1),
            np.log(abs(longer_token_length - shorter_token_length) + 1),
            1.0 * shorter_token_length / longer_token_length,
            word_difference_ratio(q1, q2),
        ]
    def cal_summary(csv_data):
        short_lt = []
        long_lt = []
        diff_lt = []
        diff_ratio_lt = []
        word_difference_ratio_lt = []
        for i in range(csv_data.shape[0]):
            a1,a2,a3,a4,a5 = extract_tokenized_features((csv_data.iloc[i, 1].split(' '), csv_data.iloc[i, 2].split(' ')))
            short_lt.append(a1)
            long_lt.append(a2)
            diff_lt.append(a3)
            diff_ratio_lt.append(a4)
            word_difference_ratio_lt.append(a5)
        return short_lt, long_lt, diff_lt, diff_ratio_lt, word_difference_ratio_lt
    short_lt, long_lt, diff_lt, diff_ratio_lt, word_difference_ratio_lt = cal_summary(test_data)
    save_data = pd.DataFrame({'short_lt':short_lt,
                              'long_lt':long_lt,
                              'diff_lt':diff_lt,
                              'diff_ratio_lt':diff_ratio_lt,
                              'word_difference_ratio_lt':word_difference_ratio_lt})
    return save_data.values

