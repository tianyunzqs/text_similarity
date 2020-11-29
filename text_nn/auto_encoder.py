# -*- coding: utf-8 -*-
# @Time        : 2020/8/28 10:20
# @Author      : tianyunzqs
# @Description : 

import tensorflow as tf


class AutoEncoder(object):
    def __init__(self,
                 embedding_size,
                 num_hidden_layer,
                 hidden_layers):
        assert num_hidden_layer == len(hidden_layers), 'num_hidden_layer not match hidden_layers'
        self.embedding_size = embedding_size
        self.num_hidden_layer = num_hidden_layer
        self.hidden_layers = hidden_layers
        self.input_x = tf.placeholder(tf.float32, [None, embedding_size])
        new_hidden_layers1 = [embedding_size] + hidden_layers + hidden_layers[::-1][1:]
        new_hidden_layers2 = hidden_layers + hidden_layers[::-1][1:] + [embedding_size]
        encoder_weights, encoder_biases, decoder_weights, decoder_biases = [], [], [], []
        for i, (hidden1, hidden2) in enumerate(zip(new_hidden_layers1, new_hidden_layers2)):
            if i < int(len(new_hidden_layers1) / 2.0):
                encoder_weights.append(tf.Variable(tf.random_normal([hidden1, hidden2])))
                encoder_biases.append(tf.Variable(tf.random_normal([hidden2])))
            else:
                decoder_weights.append(tf.Variable(tf.random_normal([hidden1, hidden2])))
                decoder_biases.append(tf.Variable(tf.random_normal([hidden2])))

        with tf.name_scope('output'):
            self.encoder_output = self.encoder_or_decode(self.input_x, encoder_weights, encoder_biases)
            self.decoder_output = self.encoder_or_decode(self.encoder_output, decoder_weights, decoder_biases)

        with tf.name_scope('loss'):
            self.loss = tf.reduce_mean(tf.pow(self.input_x - self.decoder_output, 2))

        self.global_step = tf.Variable(0, name="global_step", trainable=False)
        tf.summary.scalar('loss', self.loss)
        self.merge_summary = tf.summary.merge_all()
        self.saver = tf.train.Saver()

    @staticmethod
    def encoder_or_decode(input_data, encoder_weights, encoder_biases):
        layer_output = [input_data]
        for weight, biase in zip(encoder_weights, encoder_biases):
            layer_output.append(tf.nn.sigmoid(tf.add(tf.matmul(layer_output[-1], weight), biase)))
        return layer_output[-1]
