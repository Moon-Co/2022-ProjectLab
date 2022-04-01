#Prepare Data
import torch
import torch.nn as nn
import torch.optim as optim


from torchtext.datasets import Multi30k
from torchtext.data import Field, BucketIterator


import spacy #데이터 전처리를 위한 라이브러리
import numpy as np

import random
import math
import time

#set the random seeds

SEED = 1234

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed(SEED)
torch.backends.cudnn.deterministic = True

#making tokenizer [Good morning! -> "Good" "morning" "!"
spacy_de = spacy.load('de_core_news_sm') #for german
spacy_en = spacy.load('en_core_web_sm') #for english

def tokenize_de(text):
    #tokenize german text from a string into a list of strings and reverse
    #Why reverse? (Tbh, don't know, but distance is getting shorter)
    return [tok.text for tok in spacy_de.tokenizer(text)][::-1]

def tokenize_en(text):
    #Tokenizes English text from a string into a list of string
    return [tok.text for tok in spacy_en.tokenizer(text)]

SRC = Field(tokenize = tokenize_de, #source = germen
            init_token = '<sos>',#start of the sequence
            eos_token = '<eos>', #end of the sequence
            lower = True)

TRG = Field(tokenize = tokenize_en,  #target = english
            init_token = '<sos>',
            eos_token = '<eos>',
            lower = True)







