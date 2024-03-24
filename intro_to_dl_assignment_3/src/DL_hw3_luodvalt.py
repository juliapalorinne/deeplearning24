# -*- coding: utf-8 -*-
"""
   Introduction to Deep Learning
   Assignment 3: Sentiment Classification of Tweets on a Recurrent Neural Network using Pretrained Embeddings

   Valtteri Luodemäki & Julia Palorinne

   Credit: Data preparation pipeline adopted from https://medium.com/@sonicboom8/sentiment-analysis-torchtext-55fb57b1fab8
"""

import sys
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchtext
import spacy
import regex as re
import time
from tqdm import tqdm
from torchtext import vocab
os.chdir('C:/Users/julia/OneDrive/Tiedostot/Opiskelu/DeepLearning/intro_to_dl_assignment_3/')

import data

# Constants - Add here as you wish
N_EPOCHS = 5
EMBEDDING_DIM = 200

TRAIN_FILE = './data/sent140.train.mini.csv'
DEV_FILE   = './data/sent140.dev.csv'
TEST_FILE  = './data/sent140.test.csv'

TRAIN_BS = 32
DEV_BS   = 32
TEST_BS  = 32

# Auxilary functions for data preparation
tok = spacy.load('en_core_web_sm',disable=['parser', 'tagger', 'ner'])
def tokenizer(s): 
    return [w.text.lower() for w in tok(tweet_clean(s))]

def tweet_clean(text):
    text = re.sub(r'[^A-Za-z0-9]+', ' ', text) # remove non alphanumeric character
    text = re.sub(r'https?:/\/\S+', ' ', text) # remove links
    return text.strip()



# Evaluation functions
def evaluate(model, loader, criterion):    
    epoch_loss = 0
    epoch_acc = 0
    
    with torch.no_grad():
        for batch in loader:
            #WRITE CODE HERE
            pass

    return epoch_loss / len(loader), epoch_acc / len(loader)


# Utility
def epoch_time(start_time, end_time):
    elapsed_time = end_time - start_time
    elapsed_mins = int(elapsed_time / 60)
    elapsed_secs = int(elapsed_time - (elapsed_mins * 60))
    return elapsed_mins, elapsed_secs




# Recurrent Network
class RNN(nn.Module):
    def __init__(self):
        super().__init__()
        #WRITE CODE HERE

    def forward(self, inputs, lengths=None):
        #WRITE CODE HERE
        return 0 



if __name__ == '__main__':
        
        device = 'cuda' if torch.cuda.is_available() else 'cpu'

        print('Preparing dataset')
        train_loader, dev_loader, test_loader, glove_embeddings = data.get_dataset(
                tokenizer,
                TRAIN_FILE,
                DEV_FILE,
                TEST_FILE,
                TRAIN_BS,
                DEV_BS,
                TEST_BS,
                EMBEDDING_DIM) 

        model = None

        optimizer = None
        criterion = None

        # --- Train Loop ---
        print('Training')
        for epoch in range(N_EPOCHS):
            print(f'Epoch {epoch}')
            start_time = time.time()
            epoch_loss = 0
            epoch_acc = 0
            
            for batch in train_loader:
       		#WRITE CODE HERE
                pass
 
            train_loss, train_acc = (epoch_loss / len(train_loader), epoch_acc / len(train_loader)) 
            valid_loss, valid_acc = evaluate(model, dev_loader, criterion)
            
            end_time = time.time()
            epoch_mins, epoch_secs = epoch_time(start_time, end_time)
            
            print(f'Epoch: {epoch+1:02} | Epoch Time: {epoch_mins}m {epoch_secs}s')
            print(f'\tTrain Loss: {train_loss:.3f} | Train Acc: {train_acc*100:.2f}%')
            print(f'\t Val. Loss: {valid_loss:.3f} |  Val. Acc: {valid_acc*100:.2f}%')



#--- Test ---
print('Testing')
with torch.no_grad():
    # Here test your model using the test_loader
    pass
