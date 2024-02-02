"""
   Introduction to Deep Learning (LDA-T3114)
   Assignment 1: Sentiment Classification on a Feed-Forward Neural Network

   Valtteri Luodemäki & Julia Palorinne
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import random

#ATTENTION: If necessary, add the paths to your data_semeval.py and paths.py here:
#import sys
#sys.path.append('</path/to/below/modules>')
from data_semeval import *
from paths import data_dir


#--- hyperparameters ---

N_CLASSES = len(LABEL_INDICES)
N_EPOCHS = 10
LEARNING_RATE = 0.05
BATCH_SIZE = 1
REPORT_EVERY = 1
IS_VERBOSE = True


def make_bow(tweet, indices):
    feature_ids = list(indices[tok] for tok in tweet['BODY'] if tok in indices)
    bow_vec = torch.zeros(len(indices))
    bow_vec[feature_ids] = 1
    return bow_vec.view(1, -1)

def generate_bow_representations(data):
    vocab = set(token for tweet in data['training'] for token in tweet['BODY'])
    vocab_size = len(vocab) 
    indices = {w:i for i, w in enumerate(vocab)}
  
    for split in ["training","development.input","development.gold",
                  "test.input","test.gold"]:
        for tweet in data[split]:
            tweet['BOW'] = make_bow(tweet,indices)

    return indices, vocab_size

# Convert string label to pytorch format.
def label_to_idx(label):
    return torch.LongTensor([LABEL_INDICES[label]])


#--- model ---

class FFNN(nn.Module):
    # Feel free to add whichever arguments you like here.
    def __init__(self, vocab_size, n_classes, hidden1_n, hidden2_n):
        super(FFNN, self).__init__()
        self.hidden1 = nn.Linear(vocab_size, hidden1_n)
        self.relu1 = nn.ReLU()
        # self.hidden2 = nn.Linear(hidden1_n, hidden2_n)
        self.output = nn.Linear(hidden1_n, n_classes)

    def forward(self, x):
        x = self.hidden1(x)
        x = self.relu1(x)
        # x = self.hidden2(x)
        x = self.output(x)
        x = torch.nn.functional.log_softmax(x, dim=1)
        return x



#--- data loading ---

data = read_semeval_datasets(data_dir)
indices, vocab_size = generate_bow_representations(data)



#--- set up ---

hidden1_size = 12
hidden2_size = 3
model = FFNN(vocab_size, N_CLASSES, hidden1_size, hidden2_size) #add extra arguments here if you use
loss_function = torch.nn.NLLLoss()
optimizer = torch.optim.SGD(model.parameters(), lr=LEARNING_RATE)



#--- training ---

for epoch in range(N_EPOCHS):
    total_loss = 0
    # Generally speaking, it's a good idea to shuffle your
    # datasets once every epoch.
    random.shuffle(data['training'])

    for i in range(int(len(data['training'])/BATCH_SIZE)):
        minibatch = data['training'][i*BATCH_SIZE:(i+1)*BATCH_SIZE]
        optimizer.zero_grad()

        for tweet in minibatch:
            target = label_to_idx(tweet['SENTIMENT'])
            input = model.forward(tweet['BOW'])
            train_loss = loss_function(input, target)
            train_loss.backward()
            optimizer.step()
            total_loss += train_loss
        
        # inputs = []
        # targets = []
        # for tweet in minibatch:
        #     targets.append(label_to_idx(tweet['SENTIMENT']))
        #     inputs.append(tweet['BOW'])        

        # input_tensor = torch.stack(inputs)
        # target_tensor = torch.stack(targets)
        # output = model.forward(input_tensor)
        # train_loss = loss_function(output, target_tensor)
        # train_loss.backward()
        # optimizer.step()
        # total_loss += train_loss
        
    if ((epoch+1) % REPORT_EVERY) == 0:
        print('epoch: %d, loss: %.4f' % (epoch+1, total_loss*BATCH_SIZE/len(data['training'])))


#--- test ---

correct = 0
with torch.no_grad():
    for tweet in data['test.gold']:
        gold_class = label_to_idx(tweet['SENTIMENT'])

        predicted = torch.argmax(torch.exp(model.forward(tweet['BOW'])))
        if gold_class.item() == predicted.item():
            correct += 1
        

        if IS_VERBOSE:
           print('TEST DATA: %s, GOLD LABEL: %s, GOLD CLASS %d, OUTPUT: %d' % 
                (' '.join(tweet['BODY'][:-1]), tweet['SENTIMENT'], gold_class, predicted))

    print('test accuracy: %.2f' % (100.0 * correct / len(data['test.gold'])))
