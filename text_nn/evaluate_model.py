# -*- coding: utf-8 -*-
# @Time        : 2020/8/31 9:23
# @Author      : tianyunzqs
# @Description : 

import gzip
import json
import pickle
import jieba
import numpy as np
import tensorflow as tf
from text_nn.auto_encoder import AutoEncoder


def load_stopwords(path):
    return set([line.strip() for line in open(path, "r", encoding="utf-8").readlines() if line.strip()])


def load_word2vec_model(path):
    with gzip.open(path, 'rb') as f:
        word_vectors = pickle.load(f)
    return word_vectors


def load_config(path):
    with open(path, 'r', encoding='utf-8') as f:
        config = json.load(f)
    return config


def get_word_vector(word2vec_model_instance, word):
    assert word2vec_model_instance, 'you must load an word2vec model first'
    if word not in word2vec_model_instance.wv.vocab:
        word_vec = []
        for char in word:
            if char in word2vec_model_instance.wv.vocab:
                word_vec.append(word2vec_model_instance.wv[char])
            else:
                word_vec.append(np.random.random(word2vec_model_instance.wv.vector_size))
        if word_vec:
            return np.mean(word_vec, axis=0)
        else:
            return np.random.random(word2vec_model_instance.wv.vector_size)
    return word2vec_model_instance.wv[word]


config = load_config('./config.json')
stopwords = load_stopwords('../dict_models/stopwords.txt')
word2vec_model = load_word2vec_model(r'../dict_models/sgns.merge.char.gzip')


def load_model(path):
    with tf.Graph().as_default():
        sess = tf.Session()
        with sess.as_default():
            auto_encoder_instance = AutoEncoder(embedding_size=config['embedding_size'],
                                                num_hidden_layer=config['num_hidden_layer'],
                                                hidden_layers=config['hidden_layers'])
            auto_encoder_instance.saver.restore(sess, path)
    return sess, auto_encoder_instance


sess, model = load_model('../dict_models/auto_encoder_models/autoencoder.ckpt')


def auto_encoder_predict(texts):
    result = []
    model_input = []
    for text in texts:
        if not text:
            result.append(None)
            continue
        words = jieba.lcut(text)
        words_vec = []
        for word in words:
            word = word.strip()
            if not word and word in stopwords:
                continue
            words_vec.append(get_word_vector(word2vec_model, word))
        model_input.append(np.mean(words_vec, axis=0))
    result.extend(sess.run(model.encoder_output, feed_dict={model.input_x: model_input}))
    return result


def cosine_similarity(vec1, vec2):
    tmp1, tmp2 = np.dot(vec1, vec1), np.dot(vec2, vec2)
    if tmp1 and tmp2:
        return np.dot(vec1, vec2) / (np.sqrt(tmp1) * np.sqrt(tmp2))
    return 0.0


def compute_similarity(text1, text2):
    vec12 = auto_encoder_predict([text1, text2])
    return cosine_similarity(vec12[0], vec12[1])


if __name__ == '__main__':
    content = ["", "民警高速救助故障车辆"]
    print(auto_encoder_predict(content))
    print(compute_similarity("民警高速救助故障车辆", "民警救助故障车辆"))
