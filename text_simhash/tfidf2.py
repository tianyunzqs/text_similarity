# -*- coding: utf-8 -*-
# @Time        : 2019/6/25 14:40
# @Author      : tianyunzqs
# @Description : 


import os
import re
import codecs
import math


data_dict = dict()
for dirname, _, filenames in os.walk(r'./wiki_zh'):  # 'D:\\alg_file\\data\\wiki_zh'
    for filename in filenames:
        if re.search(r'out$', filename):
            print(os.path.join(dirname, filename))
            with codecs.open(os.path.join(dirname, filename), 'r', encoding='utf-8') as f:
                lines = f.readlines()
                for line in lines:
                    line = line.strip()
                    parts = line.split('\t')
                    if len(parts) != 3:
                        continue
                    file_key = os.path.join(dirname, filename) + parts[0]
                    if file_key not in data_dict:
                        data_dict[file_key] = {parts[1]: int(parts[2])}
                    else:
                        if parts[1] not in data_dict[file_key]:
                            data_dict[file_key][parts[1]] = int(parts[2])
                        else:
                            data_dict[file_key][parts[1]] += int(parts[2])

total_file = len(data_dict)
words_files = dict()
for file, word_freq_dict in data_dict.items():
    for word, freq in word_freq_dict.items():
        if word not in words_files:
            words_files[word] = {file}
        else:
            words_files[word].add(file)

words_idf = dict()
for word, files in words_files.items():
    words_idf[word] = math.log(total_file / (len(files) + 1))

with codecs.open('idf.txt', 'w', encoding='utf-8') as f:
    for word, idf in words_idf.items():
        f.write(word + '\t' + str(idf))
        f.write('\n')
