#! -*- coding: utf-8 -*-
"""
数据预处理
"""
import json
import math
import numpy as np
import collections

MAX_LEN = 20


class DataProcess(object):
    def __init__(self, embedding_size=200):
        self.embedding_size = embedding_size
        self.words2id = collections.defaultdict(int)
        self.id2words = collections.defaultdict(str)

    def load_vocabulary(self, path):
        vocab = self.read_txt(path)
        for index, word in enumerate(vocab):
            self.words2id[word] = index
            self.id2words[index] = word

    def load_embedding(self, path):
        embedded = np.zeros((len(self.words2id), self.embedding_size))
        embedding = self.load_json(path)
        for key, value in embedding.items():
            word_id = self.words2id.get(key, 0)
            embedded[word_id] = value
        embedded[0] = 0
        return embedded

    def tf(self, data):
        """计算词频"""
        word_tf = dict.fromkeys(self.words2id, 0)
        # word_tf['bos'] = len(data)
        # word_tf['eos'] = len(data)
        for line in data:
            line = line.strip().split()
            for word in line:
                if word in word_tf:
                    word_tf[word] += 1
        total = sum(list(word_tf.values()))
        word_tf = {k: v/total for k, v in word_tf.items()}

        return word_tf

    def idf(self, data):
        """计算逆向文件"""
        word_idf = dict.fromkeys(self.words2id, 0)
        # word_idf['bos'] = len(data)
        # word_idf['eos'] = len(data)
        for line in data:
            line = set(line.strip().split())
            for word in line:
                if word in word_idf:
                    word_idf[word] += 1

        word_idf = {k: math.log(len(data)/(v+1)) for k, v in word_idf.items()}
        # word_idf = dict(map(lambda k, v: math.log(len(data)/(v+1)), word_idf.items()))

        return word_idf

    def tf_idf(self, path):
        """计算词的tf-idf权重，训练模型时使用"""
        data = list(self.read_txt(path))
        word_tf = self.tf(data)
        word_idf = self.idf(data)
        cross_entropy_loss_weight = {}

        for word in self.words2id.keys():
            tf_idf = word_tf[word]*word_idf[word]
            if tf_idf < 0:
                tf_idf = -tf_idf
            cross_entropy_loss_weight[word] = tf_idf

        # weight = {word: word_tf[word]*word_idf[word] for word in self.words2id.keys()}

        return cross_entropy_loss_weight

    @classmethod
    def read_txt(cls, path):
        with open(path, encoding='utf-8') as f:
            lines = f.readlines()
            for line in lines:
                line = line.strip()
                yield line

    @classmethod
    def load_json(cls, path):
        with open(path, encoding='utf-8') as f:
            file = json.load(f)

        return file if file else None

    @classmethod
    def save_json(cls, file, path):
        with open(path, 'w', encoding='utf-8') as f:
            json.dump(file, f, ensure_ascii=False)


class BuildExamples(DataProcess):
    def __init__(self):
        super(BuildExamples, self).__init__()
        self.examples = collections.defaultdict(list)

    def build_example(self, path):
        data = self.read_txt(path)
        for line in data:
            line = line.strip().split()
            line = ['bos'] + line + ['eos']
            sentence2id = [self.words2id.get(item, 0) for item in line]

            if len(sentence2id) > MAX_LEN:
                # print('too long sequence!')
                continue
            self.examples['seq'].append(sentence2id)


if __name__ == '__main__':
    source_path = '/data/gump/project-data/'
    process = DataProcess()
    process.load_vocabulary(source_path+'data/vocabulary.txt')

    data = process.read_txt(source_path+'data/source.txt')
    tf = process.tf(data)
    process.save_json(tf, path=source_path+'data/tf.json')

    tf_and_idf = process.tf_idf(source_path+'data/source.txt')
    # embedding = process.load_embedding(source_path+'embedding.json')
    process.save_json(tf_and_idf, path=source_path+'data/tf_idf.json')
    # process.save_json(embedding, path=source_path+'embedded.json')
    # np.savez_compressed(source_path+'embedded.npz', embedded=embedding)
