# -*- coding: utf-8 -*-
# @Time        : 2019/11/29 17:38
# @Author      : tianyunzqs
# @Description : 文本差异性比较

import difflib


def text_repeat(string1: str, string2: str):
    """
    计算文本相似
    :param string1:
    :param string2:
    :return:
    """
    return difflib.SequenceMatcher(None, string1, string2).ratio()


if __name__ == '__main__':
    s1 = '中国的首都是北京'
    s2 = '北京是中国的首都'
    print(text_repeat(s1, s2))
