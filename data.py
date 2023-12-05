import torch
from torch.utils.data import DataLoader, Dataset
import conf
import re
import numpy as np

def tokenlize(content:str):
    # 变成小写
    content = content.lower()
    # 删除的字符
    filters = ['\t','\r', '\n','\x97','\x96','\\.','#',',','\\?','\\(','\\)','!', '"']
    for filter in filters:
        content = re.sub(filter, " ", content)
    tokens = [i.strip() for i in content.split()]
    return tokens

class Dict:
    PAD = 0
    UNK = 1
    def __init__(self):
        self.dictionary = {}
        self.addWord('PAD')
        self.addWord('UNK')

    def addWord(self, word:str):
        if word not in self.dictionary:
            self.dictionary[word] = len(self.dictionary)

    def sentence2Sequence(self, sentence)->list[int]:
        res = []
        tokens = tokenlize(sentence)
        for token in tokens:
            if token not in self.dictionary:
                res.append(self.UNK)
            else:
                res.append(self.dictionary[token])
        return res

    def __len__(self):
        return len(self.dictionary)

# 实例化词典类，全局变量
dictionary = Dict()
with open('words.txt', 'r') as f:
    lines = f.readlines()
    for word in lines:
        if word[-1] == '\n' or word[-1] == '\r':
            word = word[:-1]
            dictionary.addWord(word)
# with open(conf.data_path, 'r', encoding='utf-8', errors='ignore') as f:
#     lines = f.readlines()
#     count_dict = {}
#     for line in lines:
#         content = line[line.index(',')+1:]
#         tokens = tokenlize(content)
#         for token in tokens:
#             count_dict[token] = count_dict.get(token,0)+1
#
# for word, count in count_dict.items():
#     if count > 1:
#         dictionary.addWord(word)
#
# with open("words.txt", 'w') as f:
#     for word in dictionary.dictionary:
#         f.write(word)
#         f.write('\n')

class HamSpamDataset(Dataset):
    """
    数据量较小，一次性全部读到内存中
    20%为评估集
    """
    def __init__(self, train: bool):
        self.train = train
        self.labels = []
        self.sentences = []
        with open(conf.data_path, 'r', encoding='utf-8', errors='ignore') as file:
            lines = file.readlines()
            for line in lines:
                label = line[ : line.index(',')]
                content = line[line.index(',')+1 : ]
                self.sentences.append(content)
                self.labels.append(conf.ham if label == 'ham' else conf.spam)
        self.train_len = int(0.85*len(self.labels))
        self.eval_len = len(self.labels) - self.train_len

    def __len__(self):
        return self.train_len if self.train else self.eval_len

    def __getitem__(self, i):
        """
        :param i:
        :return: 句子的原话, 标签
        """
        if self.train:
            return self.sentences[i], self.labels[i]
        else:
            return self.sentences[i+self.train_len], self.labels[i+self.train_len]

def my_collate_fn(batch):
    """

    :param batch:
    :return:
    """
    sentences, labels = list(zip(*batch))
    sequences = []
    for sentence in sentences:
        sequence = dictionary.sentence2Sequence(sentence)
        sequences.append(torch.tensor(sequence))
    sequences = torch.nn.utils.rnn.pad_sequence(sequences, True, dictionary.PAD)

    labels = torch.tensor(labels,dtype=torch.float).unsqueeze(1)
    return sequences, labels

def getDataloader(train: bool)->DataLoader:
    """
    :param train: 是否为训练集
    :return: 格式为[]
    """
    dataset = HamSpamDataset(train)
    dataloader = DataLoader(dataset, conf.batch_size, True, collate_fn=my_collate_fn, num_workers=8)
    return dataloader

if __name__ == "__main__":
    dataloader = getDataloader(True)
    for inputs, labels in dataloader:
        print(inputs)
        print(labels)
        break