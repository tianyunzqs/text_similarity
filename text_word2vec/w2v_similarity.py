# -*- coding: utf-8 -*-
# @Time        : 2020/4/1 21:09
# @Author      : tianyunzqs
# @Description :

import os
import sys
import pickle
import gzip
import jieba
import numpy as np
from sklearn.manifold import TSNE
# from sklearn.decomposition import PCA
# from gensim.models import KeyedVectors, Word2Vec, Doc2Vec
project_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(project_path)


class TextSimilarity(object):
    def __init__(self, word2vec_model_path=None, doc2vec_model_path=None, stopwords_path=None):
        np.random.seed(2020)
        # self.word2vec_model = KeyedVectors.load_word2vec_format(word2vec_model_path) if word2vec_model_path else None
        # self.doc2vec_model = Doc2Vec.load(doc2vec_model_path) if doc2vec_model_path else None
        with gzip.open(word2vec_model_path, 'rb') as f:
            self.word2vec_model = pickle.load(f)
        self.stopwords = self.load_stopwords(stopwords_path)
        jieba.initialize()

    @staticmethod
    def load_stopwords(path):
        stopwords = set([line.strip() for line in open(path, "r", encoding="utf-8").readlines() if line.strip()])
        return stopwords

    def text_segment(self, text):
        return [word.strip() for word in jieba.lcut(text)
                if self.stopwords and word.strip() and word.strip() not in self.stopwords]

    def get_word_vector(self, word):
        assert self.word2vec_model, 'you must load an word2vec model first'
        if word not in self.word2vec_model.wv.vocab:
            return np.random.random(self.word2vec_model.wv.vector_size)
        return self.word2vec_model.wv[word]

    def get_sentence_vector(self, sentence):
        words = self.text_segment(sentence)
        words_vec = [self.get_word_vector(word) for word in words]
        return np.mean(words_vec, axis=0)

    def word_similarity(self, word1, word2):
        if word1 not in self.word2vec_model.wv.vocab:
            return '%s not in vocabulary' % word1
        if word2 not in self.word2vec_model.wv.vocab:
            return '%s not in vocabulary' % word2
        return self.word2vec_model.wv.similarity(word1, word2)

    @staticmethod
    def cosine_similarity(vec1, vec2):
        tmp1, tmp2 = np.dot(vec1, vec1), np.dot(vec2, vec2)
        if tmp1 and tmp2:
            return np.dot(vec1, vec2) / (np.sqrt(tmp1) * np.sqrt(tmp2))
        return 0.0

    def sentence_similarity_word2vec(self, sentence1, sentence2):
        sentence1 = sentence1.strip()
        sentence2 = sentence2.strip()
        if sentence1 == sentence2:
            return 1.0
        vec1 = self.get_sentence_vector(sentence1)
        vec2 = self.get_sentence_vector(sentence2)
        return self.cosine_similarity(vec1, vec2)

    # def sentence_similarity_doc2vec(self, sentence1, sentence2):
    #     sentence1 = sentence1.strip()
    #     sentence2 = sentence2.strip()
    #     if sentence1 == sentence2:
    #         return 1.0
    #     vec1 = self.doc2vec_model.infer_vector(self.text_segment(sentence1))
    #     vec2 = self.doc2vec_model.infer_vector(self.text_segment(sentence2))
    #     return self.cosine_similarity(vec1, vec2)

    def dimension_decomposition(self, text, dim=2):
        words = self.text_segment(text)
        words_vec = [self.get_word_vector(word) for word in words]
        low_dim_vec = TSNE(n_components=dim).fit_transform(words_vec)
        # low_dim_vec = PCA(n_components=dim).fit_transform(words_vec)
        return dict(zip(words, low_dim_vec))


ts = TextSimilarity(
    word2vec_model_path=os.path.join(project_path, 'dict_models', 'zhwiki_50d.word2vec.gzip'),
    stopwords_path=os.path.join(project_path, 'dict_models', 'stopwords.txt')
)


def sentence_similarity_word2vec(text1, text2):
    return ts.sentence_similarity_word2vec(text1, text2)


def dimension_decomposition(text, dim=2):
    return ts.dimension_decomposition(text, dim=dim)


if __name__ == '__main__':
    ts = TextSimilarity(word2vec_model_path=r'../dict_models/zhwiki_50d.word2vec.gzip',
                        # doc2vec_model_path=r'../dict_models/d2v.model',
                        stopwords_path=r'../dict_models/stopwords.txt')
    # print(ts.sentence_similarity_word2vec('他是一个舞蹈艺术家', '他是一个舞蹈老师'))
    # print(ts.sentence_similarity_doc2vec('他是一个舞蹈艺术家', '他是一个舞蹈老师'))
    # print(ts.dimension_decomposition('他是一个舞蹈老师'))

    print(ts.sentence_similarity_word2vec(
        '聚类是将数据分类到不同的类，所以同一个簇中的对象有很大的相似性，而不同簇间的对象有很大的相异性。',
        '聚类是将数据分类到不同的类或者簇这样的一个过程，所以同一个簇中的对象有很大的相似性'
    ))
    print(ts.dimension_decomposition('他是一个舞蹈老师'))
