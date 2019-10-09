# language-model

## 模型训练

### 数据
数据地址 

包括： 

`seg_20w.txt`: 训练数据（已分词）； 

`vocabulary.txt`: 词典文件； 

`embedded.npz`: 预训练词向量，和词典对应； 

`cross_entropy_loss_weight.json`: 交叉熵权重文件； 

'model/': 模型存放地址。

### 参数设置
参见文件：`args.py`。 

部分参数说明： 

&emsp;`embedded`: 是否引入预训练词向量。预训练词向量来源为腾讯200维词向量。 

&emsp;`weighted`: 计算损失函数时，是否加入权重。权重为词的tf-idf值。 


### 训练
`python train.py`

## 测试
测试文件: `test.py`
测试包括三部分：
1. `next_one`: 给出上下文，预测下一个词；
2. `sentence_complement`: 句子补充；
3. `ppl`: 计算句子的复杂度。
4. `get_hidden_state`: 获得句子句向量，用以计算句子间相似度
