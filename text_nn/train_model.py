# -*- coding: utf-8 -*-
# @Time        : 2020/8/28 12:01
# @Author      : tianyunzqs
# @Description : 

import os
import sys
import gzip
import pickle
import math
import random
import json
from collections import OrderedDict
import jieba
import numpy as np
import tensorflow as tf
from text_nn.auto_encoder import AutoEncoder

tf.flags.DEFINE_float('learning_rate', 0.001, 'learning rate for training model')
tf.flags.DEFINE_integer('epochs', 20, 'number of training epochs')
tf.flags.DEFINE_integer('batch_size', 32, 'batch size')
tf.flags.DEFINE_integer('num_hidden_layer', 3, 'number of auto encoder hidden layer')
tf.flags.DEFINE_list('hidden_layers', [128, 64, 32], 'number of feature for each hidden layer')
FLAGS = tf.flags.FLAGS
assert FLAGS.num_hidden_layer == len(FLAGS.hidden_layers), 'num_hidden_layer not match hidden_layers'
random.seed(2020)
np.random.seed(2020)
tf.random.set_random_seed(2020)


def load_stopwords(path):
    return set([line.strip() for line in open(path, "r", encoding="utf-8").readlines() if line.strip()])


def load_word2vec_model(path):
    with gzip.open(path, 'rb') as f:
        word_vectors = pickle.load(f)
    return word_vectors


stopwords = load_stopwords(r'../dict_models/stopwords.txt')
word2vec_model = load_word2vec_model(r'../dict_models/sgns.merge.char.gzip')


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


def load_corpus(path):
    result = []
    with open(path, 'r', encoding='utf-8') as f:
        line = f.readline()
        while line:
            line = line.strip()
            words = jieba.lcut(line)
            words_vec = []
            for word in words:
                word = word.strip()
                if not word and word in stopwords:
                    continue
                words_vec.append(get_word_vector(word2vec_model, word))
            if words_vec:
                result.append(np.mean(words_vec, axis=0))
            line = f.readline()
    return result


class BatchManager(object):
    def __init__(self, data,  batch_size):
        self.batch_data = self.gen_batch(data, batch_size)
        self.len_data = len(self.batch_data)

    @staticmethod
    def gen_batch(data, batch_size):
        num_batch = int(math.ceil(len(data) / batch_size))
        batch_data = list()
        for i in range(num_batch):
            batch_data.append(data[i*batch_size: (i+1)*batch_size])
        return batch_data

    def iter_batch(self, shuffle=False):
        if shuffle:
            random.shuffle(self.batch_data)
        for idx in range(self.len_data):
            yield self.batch_data[idx]


def save_config(data, save_path):
    config = OrderedDict()
    config['embedding_size'] = len(data[0])
    config['num_hidden_layer'] = FLAGS.num_hidden_layer
    config['hidden_layers'] = FLAGS.hidden_layers
    with open(save_path, 'w', encoding='utf-8') as f:
        json.dump(config, f, ensure_ascii=False, indent=4)


def save_model(sess, model, path):
    if os.path.isfile(path):
        raise RuntimeError('the save path should be a dir')
    if not os.path.isdir(path):
        os.makedirs(path)
    checkpoint_path = os.path.join(path, "autoencoder.ckpt")
    model.saver.save(sess, checkpoint_path)


data = load_corpus(r'D:\标注数据集\新闻标题数据集\train_label0.txt')
save_config(data, 'config.json')

with tf.Graph().as_default():
    sess = tf.Session()
    with sess.as_default():
        auto_encoder_instance = AutoEncoder(embedding_size=len(data[0]),
                                            num_hidden_layer=FLAGS.num_hidden_layer,
                                            hidden_layers=FLAGS.hidden_layers)
        optimizer = tf.train.AdamOptimizer(FLAGS.learning_rate)
        grads_and_vars = optimizer.compute_gradients(auto_encoder_instance.loss)
        train_op = optimizer.apply_gradients(grads_and_vars, global_step=auto_encoder_instance.global_step)

        train_writer = tf.summary.FileWriter('./', sess.graph)
        sess.run(tf.global_variables_initializer())
        batch_manager = BatchManager(data, FLAGS.batch_size)
        best_loss = sys.maxsize
        for epoch in range(FLAGS.epochs):
            step = 0
            for train_batch in batch_manager.iter_batch():
                _, global_step, loss, merge_summary = sess.run([train_op,
                                                                auto_encoder_instance.global_step,
                                                                auto_encoder_instance.loss,
                                                                auto_encoder_instance.merge_summary],
                                                               feed_dict={auto_encoder_instance.input_x: train_batch})
                step += 1
                train_writer.add_summary(merge_summary, global_step)
                print("epoch: ", '%d/%d' % (epoch + 1, FLAGS.epochs),
                      "\tstep: ", '%d/%d' % (step, batch_manager.len_data),
                      "\tloss: ", "{:.8f}".format(loss))
                if loss < best_loss:
                    best_loss = loss
                    save_model(sess, auto_encoder_instance, '../dict_models/auto_encoder_models')

        print("Optimization Finished!")
