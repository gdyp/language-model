#! -*- coding: utf-8 -*-
import os
import torch
import time

import torch.nn.functional as F

from args import args
from model import DNN
from utils import cos_sim
from processing import BuildExamples

# 加载模型
MODEL_PATH = os.path.join(args.get('data_path'), 'model/embedded_adam_0.001_19.model')
print(MODEL_PATH)
model = DNN(vocab_size=5265, embedding_size=200, hidden_size=512)
model.load_state_dict(torch.load(MODEL_PATH, map_location=torch.device('cpu')))

# 加载字典
corpus = BuildExamples()
corpus.load_vocabulary(path=os.path.join(args.get('data_path'), 'vocabulary.txt'))

MAX_LEN = args.get('max_len', 20)


def get_hidden_state(sentence):
    line = sentence.strip().split()
    line = ['bos'] + line + ['eos']
    sentence2id = torch.LongTensor([corpus.words2id.get(item, 0) for item in line]).view(1, -1)

    with torch.no_grad():
        last_hidden_state, _ = model(sentence2id)

    return last_hidden_state.squeeze().numpy()


def next_one(sentence):
    """根据上下文，预测下一个词"""
    line = sentence.strip().split()
    line = ['bos'] + line
    sentence2id = torch.LongTensor([corpus.words2id.get(item, 0) for item in line]).view(1, -1)

    with torch.no_grad():
        _, output = model(sentence2id)
        prob = F.softmax(output.squeeze(), dim=-1)

    max_id = prob.argmax().item()

    return corpus.id2words[max_id]


def sentence_complement(sentence):
    """句子补充"""
    while True:
        add = next_one(sentence)
        if add == 'eos':
            break
        else:
            sentence += ' {}'.format(add)
            # print(sentence)
    return sentence


def ppl(sentence):
    """计算句子复杂度"""
    line = sentence.strip().split()
    line = ['bos'] + line + ['eos']
    sentence2id = [corpus.words2id.get(item, 0) for item in line]
    length = len(sentence2id)

    if length > MAX_LEN:
        print('too long sentence!')

    ppl = 1
    for i in range(1, length-1):
        input_seq = torch.LongTensor(sentence2id[:i]).view(1, -1)
        target = torch.LongTensor([sentence2id[i+1]])

        with torch.no_grad():
            _, output = model(input_seq)  # 1*5262
            prob = F.softmax(output.squeeze(), dim=-1)

            prob = torch.index_select(prob, 0, target)

        ppl *= (1/prob.item())

    ppl = pow(ppl, 1/(length-1))

    return ppl


if __name__ == '__main__':
    # sentence = '我 喜欢 吃 火'
    # print(sentence_complement(sentence))
    s1 = '你 把 衣服 脱 光 了'
    s2 = '你 把 衣服 脱 了'
    begin = time.time()

    v1 = get_hidden_state(s1)
    print(time.time()-begin)
    v2 = get_hidden_state(s2)

    sim = cos_sim(v1, v2)
    print('{}\t与\t{}\t的相似度为：{}'.format(s1.replace(' ', ''), s2.replace(' ', ''), sim))