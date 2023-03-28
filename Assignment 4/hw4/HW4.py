#!/usr/bin/env python
# coding: utf-8

# In[14]:


import pandas as pd
import numpy as np
import csv
import torch
import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
from torch.autograd import Variable
import math
from torch.utils.data import Dataset, DataLoader, TensorDataset
device = torch.device('cpu')
import warnings
warnings.filterwarnings('ignore')

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

from torch.optim.lr_scheduler import StepLR


# # Dataset Loading from Train, Dev and Test files

# In[15]:


def prepare_train_dev_data(path):
    train_dev_df = list()
    with open(path, 'r') as f:
        for line in f.readlines():
            if len(line) > 2:
                id, word, ner_tag = line.strip().split(" ")
                train_dev_df.append([id, word, ner_tag])

    train_dev_df = pd.DataFrame(train_dev_df, columns=['id', 'word', 'NER'])
    train_dev_df = train_dev_df.dropna()
    
    X_train_dev, Y_train_dev = list(), list()
    x, y = list(), list()
    first = 1
    for row in train_dev_df.itertuples():
        if(row.id == '1' and first == 0):
            X_train_dev.append(x)
            Y_train_dev.append(y)
            x = list()
            y = list()
        first = 0
        x.append(row.word)
        y.append(row.NER)

    X_train_dev.append(x)
    Y_train_dev.append(y)
    return X_train_dev, Y_train_dev

def prepare_test_data(path):
    test_df = list()
    with open(path, 'r') as f:
        for line in f.readlines():
            if len(line) > 1:
                id, word = line.strip().split(" ")
                test_df.append([id, word])

    test_df = pd.DataFrame(test_df, columns=['id', 'word'])
    test_df = test_df.dropna()
    X_test = list()
    x = list()
    first = 1
    for row in test_df.itertuples():
        if(row.id == '1' and first == 0):
            X_test.append(x)
            x = list()
        first = 0
        x.append(row.word)

    X_test.append(x)

    return X_test


X_train, Y_train = prepare_train_dev_data('./data/train')
X_dev, Y_dev = prepare_train_dev_data('./data/dev')
X_test = prepare_test_data('./data/test')


# # Dataset Preparation

# In[16]:


'''
Sentence Vector Preparation
'''
def sentence_vector(x_train_data, word_idx):

    x_train_vector = list()
    temp = list()
    for words in x_train_data:
        for word in words:
            temp.append(word_idx[word])
        x_train_vector.append(temp)
        temp = list()

    return x_train_vector

'''
Label Vector Preparation
'''
def label_vector(y_train_data, label_dict):

    y_train_vector = list()
    for tags in y_train_data:
        temp = list()
        for label in tags:
            temp.append(label_dict[label])
        y_train_vector.append(temp)
    return y_train_vector

'''
Word Dictionary Preparation: We prepare word dictionary by setting the word as Key and its index 
in the corpus as the Value
'''
word_idx = {"<PAD>": 0, "<UNK>": 1}
idx = 2

for data in [X_train, X_dev, X_test]:
    for sentence in data:
        for word in sentence:
            if word not in word_idx:
                word_idx[word] = idx
                idx += 1

'''
Sentence Vector Preparation Driver
'''
train_x_vec = sentence_vector(X_train, word_idx)
test_x_vec = sentence_vector(X_test, word_idx)
val_x_vec = sentence_vector(X_dev, word_idx)

'''
Label Dictionary Preparation
'''    
label1 = set()
for tags_list in Y_train:
    for tag in tags_list:
        label1.add(tag)

label2 = set()
for tags_list in Y_dev:
    for tag in tags_list:
        label2.add(tag)

label = label1.union(label2)
label_tuples = []
counter = 0
for tags in label:
    label_tuples.append((tags, counter))
    counter += 1
label_dict = dict(label_tuples)

'''
Label Vector Preparation Driver
'''
train_y_vec = label_vector(Y_train, label_dict)
val_y_vec = label_vector(Y_dev, label_dict)


# In[17]:


'''
InputDataLoader purpose:
We use InputDataLoader to convert each and every vector of the train, test and dev vector to tensors
'''
class InputDataLoader(Dataset):
    def __init__(self, x, y):
        self.x = x
        self.y = y

    def __len__(self):
        return len(self.x)

    def __getitem__(self, index):
        x_instance = torch.tensor(self.x[index])
        y_instance = torch.tensor(self.y[index])
        return x_instance, y_instance

class InputTestDataLoader(Dataset):
    def __init__(self, x):
        self.x = x

    def __len__(self):
        return len(self.x)

    def __getitem__(self, index):
        x_instance = torch.tensor(self.x[index])
        return x_instance
    
'''
Collate Usage:
I have used custom Collate functionality to determine how individual samples are combined into batches during training or 
testing, since each sample in the dataset may have different sizes or shapes, they cannot be directly combined into batches.
Thus, we pad all samples/sentences to a fixed length and combines them into a tensor.
'''
class CollateData(object):

    def __call__(self, batch):
        (xx, yy) = zip(*batch)
        
        batch_max_len = float('-inf')
        for s in xx:
            batch_max_len = max(batch_max_len, len(s))
        x_len = []
        y_len = []
        for x in xx:
            x_len.append(len(x))
        for y in yy:
            y_len.append(len(y))

        batch_data = 0*np.ones((len(xx), batch_max_len))
        batch_labels = -1*np.zeros((len(xx), batch_max_len))
        for j in range(len(xx)):
            batch_data[j][:len(xx[j])] = xx[j]
            batch_labels[j][:len(xx[j])] = yy[j]

        batch_data, batch_labels = torch.LongTensor(batch_data), torch.LongTensor(batch_labels)
        batch_data, batch_labels = Variable(batch_data), Variable(batch_labels)

        return batch_data, batch_labels, x_len, y_len

class CollateTestData(object):
    def __call__(self, batch):
        xx = batch
        batch_max_len = float('-inf')
        for s in xx:
            batch_max_len = max(batch_max_len, len(s))
        x_len = []
        for x in xx:
            x_len.append(len(x))
            
        batch_data = 0*np.ones((len(xx), batch_max_len))
        batch_labels = -1*np.zeros((len(xx), batch_max_len))
        for j in range(len(xx)):
            batch_data[j][:len(xx[j])] = xx[j]

        batch_data = torch.LongTensor(batch_data)
        batch_data = Variable(batch_data)

        return batch_data, x_len


# # Task 1: Simple Bidirectional LSTM model

# In[18]:


'''
Task-1
Simple BiLSTM Model for the Task - 1
- embedding dim =  100
- number of LSTM layers =  1
- LSTM hidden dim =  256
- LSTM Dropout =  0.33
- Linear output dim =  128
'''
class Custom_network(nn.Module):
    def __init__(self, vocab_size, embedding_dim, linear_out_dim, hidden_dim, lstm_layers,
                 dropout_val, tag_size):
        super(Custom_network, self).__init__()
        self.embedding_dim = embedding_dim
        self.lstm_layers = lstm_layers
        self.hidden_dim = hidden_dim
        self.linear_out_dim = linear_out_dim
        self.tag_size = tag_size
        self.num_directions = 2

        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.embedding.weight.data.uniform_(-1,1)
        self.LSTM = nn.LSTM(embedding_dim, hidden_dim, num_layers=lstm_layers, batch_first=True, bidirectional=True)
        self.fc = nn.Linear(hidden_dim*self.num_directions, linear_out_dim)
        self.classifier = nn.Linear(linear_out_dim, self.tag_size)
        self.dropout = nn.Dropout(dropout_val)
        self.elu = nn.ELU()

    def init_hidden(self, batch_size):
        h, c = (torch.zeros(self.lstm_layers * self.num_directions, batch_size, self.hidden_dim).to(device),
                torch.zeros(self.lstm_layers * self.num_directions, batch_size, self.hidden_dim).to(device))
        return h, c

    def forward(self, sen, sen_len):
        batch_size = sen.shape[0]
        h_0, c_0 = self.init_hidden(batch_size)

        embedded = self.embedding(sen).float()
        packed_embedded = pack_padded_sequence(embedded, sen_len, batch_first=True, enforce_sorted=False)
        output, _ = self.LSTM(packed_embedded, (h_0, c_0))
        output_unpacked, _ = pad_packed_sequence(output, batch_first=True)
        dropout = self.dropout(output_unpacked)
        lin = self.fc(dropout)
        pred = self.elu(lin)
        pred = self.classifier(pred)
        return pred


# In[19]:


# Training Data
model_custom = Custom_network(vocab_size=len(word_idx),embedding_dim=100,linear_out_dim=128,
                      hidden_dim=256, lstm_layers=1,dropout_val=0.33,tag_size=len(label_dict))
model_custom.to(device)
print(model_custom)

train_loader_input = InputDataLoader(train_x_vec, train_y_vec)
custom_collator = CollateData()
dataloader = DataLoader(dataset=train_loader_input, batch_size=5, drop_last=True, collate_fn=custom_collator)
criterion = nn.CrossEntropyLoss(ignore_index=-1)
criterion = criterion.to(device)
criterion.requres_grad = True
optimizer = torch.optim.SGD(model_custom.parameters(), lr=0.1, momentum=0.9)
epochs = 200 #200

for i in range(1, epochs+1):
    train_loss = 0.0
    for input, label, input_len, label_len in dataloader:
        optimizer.zero_grad()
        output = model_custom(input.to(device), input_len)
        output = output.view(-1, len(label_dict))
        label = label.view(-1)
        loss = criterion(output, label.to(device))
        loss.backward()
        optimizer.step()
        train_loss += loss.item() * input.size(1)
        
    train_loss = train_loss / len(dataloader.dataset)
    print('Epoch: {} \tTraining Loss: {:.6f}'.format(i, train_loss))
    
with torch.no_grad():
    for input, label, input_len, label_len in dataloader:
        optimizer.zero_grad()
        output = model_custom(input.to(device), input_len)
        output = output.view(-1, len(label_dict))
        label = label.view(-1)
        loss = criterion(output, label.to(device))
        optimizer.step()
    torch.save(model_custom.state_dict(), 'blstm1.pt')


# In[29]:


# Development Data
dev_loader_input = InputDataLoader(val_x_vec, val_y_vec)
custom_collator = CollateData()
dataloader_dev = DataLoader(dataset=dev_loader_input, batch_size=32, shuffle=False, drop_last=True, collate_fn=custom_collator)

label_dict_temp = {}
vocab_dict_temp = {}

for k, v in label_dict.items():
    label_dict_temp[v] = k
for k, v in word_idx.items():
    vocab_dict_temp[v] = k
    
for e in range(1,epochs + 1):
    model_custom.load_state_dict(torch.load("./blstm1.pt"))#125
    model_custom.to(device)

    file = open("./dev1_temp.out", 'w')
    file_dev_final = open("./dev1.out", 'w')
    for dev_data, label, dev_data_len, label_data_len in dataloader_dev:

        pred = model_custom(dev_data.to(device), dev_data_len)
        pred = pred.cpu()
        pred = pred.detach().numpy()
        label = label.detach().numpy()
        dev_data = dev_data.detach().numpy()
        pred = np.argmax(pred, axis=2)
        pred = pred.reshape((len(label), -1))

        for i in range(len(dev_data)):
            for j in range(len(dev_data[i])):
                if dev_data[i][j] != 0:
                                                    # word, gold, op
                    file.write(" ".join([str(j+1), str(vocab_dict_temp[dev_data[i][j]]), 
                                         str(label_dict_temp[label[i][j]]), str(label_dict_temp[pred[i][j]])]))
                    
                                                    # word, op
                    file_dev_final.write(" ".join([str(j+1), str(vocab_dict_temp[dev_data[i][j]]), 
                                                   str(label_dict_temp[pred[i][j]])]))
                    file.write("\n")
                    file_dev_final.write("\n")
            file.write("\n")
            file_dev_final.write("\n")
    file.close()
    file_dev_final.close()
    
        
with torch.no_grad():
    for input, label, input_len, label_len in dataloader_dev:
        optimizer.zero_grad()
        output = model_custom(input.to(device), input_len)
        output = output.view(-1, len(label_dict))
        label = label.view(-1)
        loss = criterion(output, label.to(device))
        optimizer.step()
        
    get_ipython().system('perl conll03eval.txt < dev1_temp.out')


# In[21]:


"""Testing on Testing Dataset """
test_loader_input = InputTestDataLoader(test_x_vec)
custom_test_collator = CollateTestData()
dataloader_test = DataLoader(dataset=test_loader_input, batch_size=32, shuffle=False, drop_last=True, 
                             collate_fn=custom_test_collator)

label_dict_temp = {}
vocab_dict_temp = {}

for k, v in label_dict.items():
    label_dict_temp[v] = k
for k, v in word_idx.items():
    vocab_dict_temp[v] = k
    
for e in range(1,epochs + 1):
    model_custom.load_state_dict(torch.load("./blstm1.pt"))#125
    model_custom.to(device)
    

    file = open("test1.out", 'w')
    for test_data, test_data_len in dataloader_test:

        pred = model_custom(test_data.to(device), test_data_len)
        pred = pred.cpu()
        pred = pred.detach().numpy()
        test_data = test_data.detach().numpy()
        pred = np.argmax(pred, axis=2)
        pred = pred.reshape((len(test_data), -1))
        
        for i in range(len(test_data)):
            for j in range(len(test_data[i])):
                if test_data[i][j] != 0:
                    word = vocab_dict_temp[test_data[i][j]]
                    op = label_dict_temp[pred[i][j]]
                    file.write(" ".join([str(j+1), word, op]))
                    file.write("\n")

            file.write("\n")        
    file.close()


# # Task 2: Using GloVe word embeddings

# In[22]:


'''
Task-2
BiLSTM Model for Task 2: Using GloVe word embeddings
- embedding dim =  100
- number of LSTM layers =  1
- LSTM hidden dim =  256
- LSTM Dropout =  0.33
- Linear output dim =  128
'''
class Glove_network(nn.Module):
    def __init__(self, vocab_size, embedding_dim, linear_out_dim, hidden_dim, lstm_layers,dropout_val, tag_size, emb_matrix):
        super(Glove_network, self).__init__()
        self.embedding_dim = embedding_dim
        self.lstm_layers = lstm_layers
        self.hidden_dim = hidden_dim
        self.linear_out_dim = linear_out_dim
        self.tag_size = tag_size
        self.emb_matrix = emb_matrix
        self.num_directions = 2

        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.embedding.weight = nn.Parameter(torch.tensor(emb_matrix))
        self.LSTM = nn.LSTM(embedding_dim, hidden_dim, num_layers=lstm_layers, batch_first=True, bidirectional=True)
        self.fc = nn.Linear(hidden_dim*self.num_directions, linear_out_dim) 
        self.classifier = nn.Linear(linear_out_dim, self.tag_size)
        self.dropout = nn.Dropout(dropout_val)
        self.elu = nn.ELU()

    def init_hidden(self, batch_size):
        h, c = (torch.zeros(self.lstm_layers * self.num_directions, batch_size, self.hidden_dim).to(device),
                torch.zeros(self.lstm_layers * self.num_directions, batch_size, self.hidden_dim).to(device))
        return h, c

    def forward(self, sen, sen_len):
        batch_size = sen.shape[0]
        h_0, c_0 = self.init_hidden(batch_size)

        embedded = self.embedding(sen).float()
        packed_embedded = pack_padded_sequence(embedded, sen_len, batch_first=True, enforce_sorted=False)
        output, _ = self.LSTM(packed_embedded, (h_0, c_0))
        output_unpacked, _ = pad_packed_sequence(output, batch_first=True)
        dropout = self.dropout(output_unpacked)
        lin = self.fc(dropout)
        pred = self.elu(lin)
        pred = self.classifier(pred)
        return pred


# In[23]:


'''
Creating Embedding Matrix for GloVe
'''
glove = pd.read_csv('./glove.6B.100d.txt', sep=" ", quoting=3, header=None, index_col=0)

glove_emb = {}
for k, v in glove.T.items():
    glove_emb[k] = v.values

    
glove_emb_list = []
for k in glove_emb:
    glove_emb_list.append(glove_emb[k])    
glove_vec = np.array(glove_emb_list)

glove_emb["<UNK>"] = np.mean(glove_vec, axis=0, keepdims=True).reshape(100,)
glove_emb["<PAD>"] = np.zeros((100,), dtype="float64")

emb_matrix = np.zeros((len(word_idx), 100))
for word_key, index_value in word_idx.items():
    if word_key not in glove_emb:
        if word_key.lower() not in glove_emb:
            emb_matrix[index_value] = glove_emb["<UNK>"]
        else:
            emb_matrix[index_value] = glove_emb[word_key.lower()] + 5e-3
    else:
        emb_matrix[index_value] = glove_emb[word_key]
            


# In[24]:


# Training Data
model_glove = Glove_network(vocab_size=len(word_idx),
                      embedding_dim=100,
                      linear_out_dim=128,
                      hidden_dim=256,
                      lstm_layers=1,
                      dropout_val=0.33,
                      tag_size=len(label_dict),
                      emb_matrix=emb_matrix)
model_glove.to(device)
print(model_glove)

train_loader_input_glove = InputDataLoader(train_x_vec, train_y_vec)
custom_collator = CollateData()
dataloader = DataLoader(dataset=train_loader_input_glove, batch_size=32, drop_last=True, collate_fn=custom_collator)

criterion = nn.CrossEntropyLoss(ignore_index=-1)
criterion = criterion.to(device)
criterion.requres_grad = True
optimizer = torch.optim.SGD(model_glove.parameters(), lr=0.1, momentum=0.9)
epochs = 220 #220

for i in range(1, epochs+1):
    train_loss = 0.0
    for input, label, input_len, label_len in dataloader:
        optimizer.zero_grad()
        output = model_glove(input.to(device), input_len)
        output = output.view(-1, len(label_dict))
        label = label.view(-1)
        loss = criterion(output, label.to(device))
        loss.backward()
        optimizer.step()
        train_loss += loss.item() * input.size(1)

    train_loss = train_loss / len(dataloader.dataset)
    print('Epoch: {} \tTraining Loss: {:.6f}'.format(i, train_loss))
    
with torch.no_grad():
    for input, label, input_len, label_len in dataloader:
        optimizer.zero_grad()
        output = model_glove(input.to(device), input_len)
        output = output.view(-1, len(label_dict))
        label = label.view(-1)
        loss = criterion(output, label.to(device))
        optimizer.step()
    torch.save(model_glove.state_dict(), 'blstm2.pt')


# In[25]:


#predicting for Development dataset
dev_loader_input = InputDataLoader(val_x_vec, val_y_vec)
custom_collator = CollateData()
dataloader_dev = DataLoader(dataset=dev_loader_input, batch_size=32, shuffle=False, drop_last=True, collate_fn=custom_collator)

model_glove.load_state_dict(torch.load("./blstm2.pt"))

label_dict_temp = {}
vocab_dict_temp = {}

for k, v in label_dict.items():
    label_dict_temp[v] = k
for k, v in word_idx.items():
    vocab_dict_temp[v] = k
    
for e in range(1, epochs+1):
    model_glove = Glove_network(vocab_size=len(word_idx), embedding_dim=100,linear_out_dim=128,hidden_dim=256,
                        lstm_layers=1,dropout_val=0.33,tag_size=len(label_dict),emb_matrix = emb_matrix)

    model_glove.load_state_dict(torch.load("./blstm2.pt"))
    model_glove.to(device)
    
    file = open("dev2_temp.out", 'w')
    file_dev2_final = open("dev2.out", 'w')
    for dev_data, label, dev_data_len, label_data_len in dataloader_dev:

        pred = model_glove(dev_data.to(device), dev_data_len)
        pred = pred.cpu()
        pred = pred.detach().numpy()
        label = label.detach().numpy()
        dev_data = dev_data.detach().numpy()
        pred = np.argmax(pred, axis=2)
        pred = pred.reshape((len(label), -1))

        for i in range(len(dev_data)):
            for j in range(len(dev_data[i])):
                if dev_data[i][j] != 0:
                        # word, gold, op
                    file.write(" ".join([str(j+1), vocab_dict_temp[dev_data[i][j]], 
                                         label_dict_temp[label[i][j]], label_dict_temp[pred[i][j]]]))
                    
                    file_dev2_final.write(" ".join([str(j+1), vocab_dict_temp[dev_data[i][j]], 
                                         label_dict_temp[pred[i][j]]]))
                    file.write("\n")
                    file_dev2_final.write("\n")
            file.write("\n")
            file_dev2_final.write("\n")
    file.close()
    file_dev2_final.close()

     
with torch.no_grad():
    for input, label, input_len, label_len in dataloader_dev:
        optimizer.zero_grad()
        output = model_glove(input.to(device), input_len)
        output = output.view(-1, len(label_dict))
        label = label.view(-1)
        loss = criterion(output, label.to(device))
        optimizer.step()
        
    get_ipython().system('perl conll03eval.txt < dev2_temp.out')


# In[26]:


# Testing Dataset
test_loader_input = InputTestDataLoader(test_x_vec)
custom_test_collator = CollateTestData()
dataloader_test = DataLoader(dataset=test_loader_input,batch_size=32, shuffle=False, drop_last=True,
                             collate_fn=custom_test_collator)
label_dict_temp = {}
vocab_dict_temp = {}

for k, v in label_dict.items():
    label_dict_temp[v] = k
for k, v in word_idx.items():
    vocab_dict_temp[v] = k
    
for e in range(1,epochs + 1):
    model_glove.load_state_dict(torch.load("./blstm2.pt"))
    model_glove.to(device)

    file = open("test2.out", 'w')
    for test_data, test_data_len in dataloader_test:

        pred = model_glove(test_data.to(device), test_data_len)
        pred = pred.cpu()
        pred = pred.detach().numpy()
        test_data = test_data.detach().numpy()
        pred = np.argmax(pred, axis=2)
        pred = pred.reshape((len(test_data), -1))
        
        for i in range(len(test_data)):
            for j in range(len(test_data[i])):
                if test_data[i][j] != 0:
                    word = vocab_dict_temp[test_data[i][j]]
                    op = label_dict_temp[pred[i][j]]
                    file.write(" ".join([str(j+1), word, op]))
                    file.write("\n")

            file.write("\n")        
    file.close()

