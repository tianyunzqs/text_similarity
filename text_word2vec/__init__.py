# -*- coding: utf-8 -*-
# @Time        : 2020/8/18 16:55
# @Author      : tianyunzqs
# @Description : 

import gzip
import jieba
import pickle
import numpy as np
# from gensim.models import KeyedVectors

# with gzip.open(r'E:\pyworkspace\NLPsat\dict_models\zhwiki_50d.word2vec.gzip', 'rb') as f:
# with gzip.open(r'sgns.merge.char.gzip', 'rb') as f:
#     word_vectors = pickle.load(f)
#
# print(word_vectors.wmdistance('同一个簇中的对象有很大的相似性', '不同簇间的对象有很大的相异性'))
# word_vectors = KeyedVectors.load_word2vec_format(r'D:\标注数据集\sgns.merge.char')
# print(word_vectors.wv['中国'])
# with gzip.open('sgns.merge.char.gzip', 'wb') as f:
#     pickle.dump(word_vectors, f)


def cos_dis(v1, v2):
    return np.linalg.norm(v1-v2)


def WMD_core(p1, p2):
    p1_vect = []
    # if can't find corresponding word embeddings, add single word embeddings
    # 分词后的词语若无对应词向量，则添加该单字向量
    for word in list(jieba.cut(p1)):
        if word in word_vectors.vocab:
            p1_vect.append(word_vectors[word])
        else:
            p1_vect.extend(map(lambda x: word_vectors[x], word))

    p2_vect = []
    for word in list(jieba.cut(p2)):
        if word in word_vectors.vocab:
            p2_vect.append(word_vectors[word])
        else:
            p2_vect.extend(map(lambda x: word_vectors[x], word))

    total_min = []
    sum_min = 0.0
    for w1 in p1_vect:
        cur_min = 1000.0
        for w2 in p2_vect:
            temp = cos_dis(w1, w2)
            if temp < cur_min:
                cur_min = temp
        total_min.append(cur_min)
        sum_min += cur_min
        print(total_min)
    return sum_min


def WMD(p1,p2_list):
    sum_res = []
    for p2 in p2_list:
        if p2:
            sum_res.append(WMD_core(p1,p2))
    return sum_res


# if __name__ == '__main__':
#     aa = '重庆必吃'
#     bb = ["婚庆公司商家特色","创战纪主题失重餐厅","不满意重拍","土豪必去","重庆菜","重庆火锅","天津必吃","旅行必吃","婚庆公司擅长风格","西安必吃","重庆小面","苏州必吃","重庆老火锅","地道重庆火锅","广州必吃","成都必吃","上海必吃","南京必吃","武汉必吃","北京必吃"]
#     a = WMD(aa, bb)
#     print(sorted(list(zip(bb, a)), key=lambda x: x[1])[:5])
#     print(sorted(list(zip(bb, a)), key=lambda x: x[1], reverse=True)[:5])
