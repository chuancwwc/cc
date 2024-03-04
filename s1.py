#!/usr/bin/env python
# coding: utf-8

import pandas as pd
import transformers
from transformers import DistilBertModel, DistilBertTokenizer
import numpy as np
import json
from sklearn.feature_extraction.text import CountVectorizer
from torch.utils.data import DataLoader, Dataset
from torch.optim import AdamW
from torch.nn import CrossEntropyLoss,BCELoss,MSELoss,BCEWithLogitsLoss,CosineEmbeddingLoss
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm
from torch import cuda



tr_path = '../wos/wos_train.json'
bert_path = '../../distilbert-base-uncased'
hieidx_path = '../wos/wos_hieidx.json'
childlist_path = '../wos/wos_childlist.json'

df_tr = pd.read_json(tr_path,lines=True)
with open(hieidx_path, 'r') as f:
    hieidx = json.load(f)
with open(childlist_path, 'r') as f:
    childlist = json.load(f)

labels=df_tr['labels']
labels=np.array(labels.tolist())
labels = labels.tolist()

def get_concatenated_values(input_list, data):
    return ''.join(data[str(i)] for i in input_list)

label_pa = []
for i in range(len(labels)):
    label = labels[i]
    result = get_concatenated_values(label, hieidx)
    label_pa.append(result)

content_train = df_tr['content']

y = []
contents = []
label_path = []

for i in range(len(content_train)):
    for n in range(7):
        if n == labels[i][0]:
            y1 = 1
            y.append(y1)
            cont = content_train[i]
            contents.append(cont)
            label1 = label_pa[i]
            label_path.append(label1)
            for m in childlist[str(n)]:
                cont = content_train[i]
                contents.append(cont)
                label1 = label_pa[i]
                label_path.append(label1)
                if m == labels[i][1]:
                    y1 = 1
                else:
                    y1 = 0
                y.append(y1)
        else:
            y1 = 0
            y.append(y1)
            cont = content_train[i]
            contents.append(cont)
            label1 = label_pa[i]
            label_path.append(label1)

#contents, label_path,y


sciModel = DistilBertModel.from_pretrained(bert_path)
tokenizer = DistilBertTokenizer.from_pretrained(bert_path)

class TextDatasetWithLabels(Dataset):
    def __init__(self, contents, label_path,y, tokenizer, max_length=512):
        self.tokenizer = tokenizer
        self.texts1 = contents
        self.texts2 = label_path
        self.labels = y
        self.max_length = max_length

    def __len__(self):
        return len(self.texts1)

    def __getitem__(self, idx):

        #print(self.texts[idx][0])
        #id1 = ''.join(self.texts[idx])
        #id2 = ''.join(self.texts[idx])
        text1 = ' '.join(self.texts1[idx])
        text2 = ' '.join(self.texts2[idx])
        label = self.labels[idx]
        #print(text1,len(text1))

        input1 = tokenizer(text1, padding='max_length', truncation=True, return_tensors="pt", max_length=self.max_length)
        input2 = tokenizer(text2, padding='max_length', truncation=True, return_tensors="pt", max_length=self.max_length)
        input_ids1 = input1['input_ids']
        attention_mask1 = input1['attention_mask']
        input_ids2 = input2['input_ids']
        attention_mask2 = input2['attention_mask']

        #print(input_ids1.shape,input_ids2.shape)

        return {
            'input1': input_ids1.squeeze(0),  # 移除批次维度
            'attention_mask1': attention_mask1.squeeze(0),
            'input2': input_ids2.squeeze(0),  # 移除批次维度
            'attention_mask2': attention_mask2.squeeze(0),
            'labels': torch.tensor(label, dtype=torch.long),  # 假设标签是整数类型
        }








dataset = TextDatasetWithLabels(contents, label_path,y, tokenizer)

loader = DataLoader(dataset, batch_size=2, shuffle=True)




class CustomModel(nn.Module):
    def __init__(self, embedding_dim, hidden_dim, output_dim):
        super(CustomModel, self).__init__()
        self.sciModel = DistilBertModel.from_pretrained(bert_path)

        self.fc1 = nn.Linear(embedding_dim*2, hidden_dim)
        self.fc2 = nn.Linear(embedding_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, output_dim)

    def forward(self, x1,x2,attention1,attention2):

        outputs1 = self.sciModel(x1,attention_mask=attention1) #content token_id
        outputs2 = self.sciModel(x2,attention_mask=attention2)

        embeddings1 = outputs1.last_hidden_state
        representations1 = embeddings1[:, 0, :] #content的句子嵌入（batch_size，features）

        embeddings2 = outputs2.last_hidden_state
        representations2 = embeddings2[:, 0, :] #label_path的句子嵌（batch_size，features）

        feature = torch.concat([representations1,representations2],dim=-1) #（batch_size，features*2）
        #x1 = F.relu(self.fc1(representations1))
        #x2 = F.relu(self.fc2(representations2))
        #print(x.shape)
        x = F.relu(self.fc1(feature))
        x = F.relu(self.fc3(x))
        #print(x.shape)
        return x






torch.backends.cudnn.enable =True
torch.backends.cudnn.benchmark = True

margin = 2
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


epochs = 2 # 训练轮数
loss_fn = BCELoss()
#loss_fn = ContrastiveLoss(margin=margin)


#loss_fn = MSELoss()
model = CustomModel(768, 300, 1).cuda()
#model = SiameseNet(768, 300).cuda()
optimizer = AdamW(model.parameters(), lr=1e-4)
n = 0
for epoch in range(epochs):
    model.train()
    total_loss = 0
    for batch in tqdm(loader):
        optimizer.zero_grad()


        #print(batch[0].shape)
        #print(batch[0])
        input1 = batch['input1'].to(device)
        input2 = batch['input2'].to(device)
        attention_mask1 = batch['attention_mask1'].to(device)
        attention_mask2 = batch['attention_mask1'].to(device)
        labels = batch['labels']

        dis = model(input1,input2,attention_mask1,attention_mask2).to(device)
        loss = loss_fn(dis.squeeze(), labels.float())
        #loss = loss_fn(outputs1.squeeze(),outputs2.squeeze(), labels.float())
        if epoch >=2:
            print(labels.float())
            print(dis)

            loss.backward()


optimizer.step()

total_loss += loss.item()


print(f"Epoch {epoch+1}/{epochs} Loss: {total_loss/len(loader)}")







