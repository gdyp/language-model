#! -*- coding: utf-8 -*-
"""
模型训练
"""
import os
import math
from tqdm import tqdm
import logging
import torch
import torch.nn as nn
from tensorboardX import SummaryWriter

from args import args
from model import DNN
from utils import padding
from processing import BuildExamples

logging.basicConfig(level='INFO', format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logging.info(args)

source_path = args.get('data_path')
os.environ["CUDA_VISIBLE_DEVICES"] = "1"
write = SummaryWriter()
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

corpus = BuildExamples()

corpus.load_vocabulary(path=os.path.join(args.get('data_path'), 'vocabulary.txt'))
corpus.build_example(path=os.path.join(args.get('data_path'), 'seg_20w.txt'))

embedding = corpus.load_embedding(path=os.path.join(args.get('data_path'), 'embedding.json'))
embedding = torch.from_numpy(embedding).float()

if args['weighted']:
    weight = corpus.load_json(os.path.join(source_path, 'cross_entropy_loss_weight.json'))
    weight = torch.FloatTensor(list(weight.values())).to(device)

data = sorted(corpus.examples.get('seq'), key=lambda x: len(x), reverse=True)

vocab_size = len(corpus.words2id)
logging.info('vocabulary size: {}'.format(vocab_size))
model = DNN(vocab_size=vocab_size, embedding_size=200, hidden_size=512, embedding=embedding)

model.to(device)

loss_function = nn.CrossEntropyLoss(weight=weight)

optimizer = torch.optim.SGD(model.parameters(), lr=0.01, momentum=0.9)
# optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

model.train()

total_data = len(data)
batch_size = args['batch_size']
total_step = math.ceil(total_data / batch_size)
last_training_loss = 1000000000000
for epoch in range(args.get('epoch')):
    start = 0
    training_loss = 0
    for _ in tqdm(range(int(total_step)), total=total_step):
        batch = data[start:start+batch_size]
        start += batch_size

        max_len, seq = padding(batch)  # list
        seq = torch.LongTensor(seq).to(device)

        for j in range(1, max_len):
            input_seq = seq[:, :j]
            target = seq[:, j]

            _, prob = model(input_seq)

            loss = loss_function(prob, target)
            training_loss += loss.item()
            loss.backward()

        optimizer.step()
        model.zero_grad()

    # write.add_scalar('weighted_embedded_sgd_0.01', training_loss/total_data, epoch)
    logging.info('epoch: {}, training loss: {}'.format(epoch, training_loss/total_data))

    if training_loss < last_training_loss:
        save_path = source_path+'/model/weighted_embedded_sgd_0.01_{}.model'.format(str(epoch))
        # torch.save(model.state_dict(), save_path)
        logging.info('saving model at {}'.format(save_path))
        last_training_loss = training_loss
