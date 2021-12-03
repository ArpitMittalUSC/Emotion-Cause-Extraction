# -*- coding: utf-8 -*-
"""model.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1tbRJCq6o4gMvdmUNSE-_z2IqyOvm61vA
"""

# !pip install benepar
# !pip install pycorenlp

import pandas as pd
import numpy as np
import torch.nn as nn
from torch.optim import SGD
import random
import warnings
import re
warnings.filterwarnings('ignore')
from torch import save
import torch
from torch.utils.data import DataLoader, Dataset
from torch.utils.data.sampler import SubsetRandomSampler
from torch import load as model_load, FloatTensor, LongTensor, tensor as Tensor
from torch import max as output_max
from torch.nn.utils.rnn import pad_sequence, pack_padded_sequence, pad_packed_sequence
import gc
import nltk
from nltk import TweetTokenizer
nltk.download('wordnet', quiet=True)
import json
import benepar   
from pycorenlp import *
import pickle
import gensim
import gensim.downloader as gensim_api
from gensim.models import Word2Vec, KeyedVectors

cuda = torch.device('cuda:0')

seed = 39542
np.random.seed(seed)
random.seed(seed)
torch.manual_seed(seed);

## ALL DATA CLEANING FUNCTIONS
def RemoveHTMLAndURLs(text):
    return re.sub("([^\s]+://)?([^\s]+\.)+[^\s]+", "", re.sub(r'<.*?>', '', text))
def RemoveExtraSpaces(text):
    return re.sub("\s+", " ", text)
def RemoveHashtags(text):
    return re.sub("#[^\s]+", "", text)
def ExpandContractions(text):
    text = re.sub("won't", "will not", text)
    text = re.sub("can't", "cannot", text)
    text = re.sub("shan't", "shall not", text)
    text = re.sub("(?:[a-z])'re", " are", text)
    text = re.sub("(?:[a-z])'d", " would", text)
    text = re.sub("(?:[a-z])'ll", " will", text)
    text = re.sub("(?:[a-z])'ve", " have", text)
    text = re.sub("n't", " not", text)
    text = re.sub("o'clock", "of the clock", text)
    return text
def RemoveSpecialChar(text):
    return re.sub("[^A-Za-z\s]", "", text)
def TrimWords(words, max_word_size):
    return [word[:max_word_size] for word in words]

def tokenize_clauses(clauses, tokenize_fn, max_word_size=20):
    tokenized = []
    for clause in clauses:
        tokenized.append(TrimWords(tokenize_fn(RemoveSpecialChar(ExpandContractions(RemoveHashtags((RemoveExtraSpaces(RemoveHTMLAndURLs(clause))))))),
                                                              max_word_size))
    return tokenized

def get_vocab(corpus):
    word_list = ['<PAD>', '<UNK>']
    for clause_it in corpus:
        word_list.extend(np.concatenate(clause_it))
    return np.unique(word_list)

def encode_words(clauses, vocab, max_clause_size=0, pad=False):
    encoded = []
    for clause in clauses:
        encoded_clause = [ np.argmax(vocab == word) if word in vocab else 1 for word in clause ] 
        # pad to max_clause_size
        if pad:
            encoded_clause.extend([0] * (max_clause_size - len(encoded_clause)))
        encoded.append(encoded_clause)
    return encoded

def pad_with_clauses(corpus, num_words, max_clauses):
    for doc in corpus:
        for _ in range(max_clauses - len(doc)):
            doc.append([0] * num_words)

def unpack_embeddings(embedding_dict, embedding_dim, vocab, wv=None):
    unknown_embedding = np.random.randn(embedding_dim)
    embeddings = []
    for word in vocab:
        if word == '<PAD>':
            embeddings.append([0] * embedding_dim)
        elif word == '<UNK>':
            embeddings.append(unknown_embedding)
        else:
            try:
                embeddings.append(embedding_dict[word])
            except KeyError:
                # print("We probably shouldn't be here.")
                if wv:
                    try:
                        embeddings.append(wv[word])
                    except KeyError:
                        pass
                embeddings.append(unknown_embedding)
                
    return embeddings

def doc_to_embeddings(doc, embedding_matrix):
    return np.array([np.mean([embedding_matrix[word] for word in clause], axis=0) if clause else embedding_matrix[0] for clause in doc])

class DFDataset(Dataset):
    def __init__(self, df):
        self.df = df

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        return FloatTensor(row["input"]), Tensor(row["label"])

tokenize = TweetTokenizer().tokenize

with open('dataset.pkl', 'rb') as file:
    raw_data = pickle.load(file)
    train_df = pd.DataFrame(raw_data).rename(columns={'Clauses':'clauses', 'causeClauseIndex':'cause'})

max_num_clauses = max(len(doc) for doc in train_df["clauses"])
num_labels = 8
max_word_size = 20   # this is passed into TrimWords to avoid super long words but probably not relevant in our dataset

train_df.loc[:, "clauses"] = train_df["clauses"].apply(lambda x: tokenize_clauses(x, tokenize, max_word_size))
vocab = get_vocab(train_df["clauses"])

train_df.loc[:, "emotion"] = train_df["emotion"].apply(int)
train_df.loc[:, "cause"] = train_df["cause"].apply(int)

pretrained_wv = gensim_api.load("glove-twitter-25")

with open('embeddings.pkl', 'rb') as file:
    trained_wv = pickle.load(file)

pretrained_embedding_matrix = unpack_embeddings(pretrained_wv, 25, vocab)
trained_embedding_matrix = unpack_embeddings(trained_wv, 300, vocab)

data_pretrained = train_df
data_trained = train_df.copy()

pretrained_emotion_set = DFDataset(data_pretrained[["review_text_we_embeddings", "emotion"]].rename(columns={'review_text_we_embeddings' : 'input', 'emotion': 'label'}))
trained_emotion_set = DFDataset(data_trained[["review_text_eae_embeddings", "emotion"]].rename(columns={'review_text_eae_embeddings' : 'input', 'emotion': 'label'}))

#model dimensions / parameters
hidden_dim = 256
linear_dim = 80
drop_in = 0.2  # dropout layer percentages
drop_out = 0.5

# hyperparamters
batch_size = 1
num_epochs = 100
lr = 0.003
momentum = 0.9

def trainAndEvalBiLSTM(model, optimizer, data, num_clauses, batch_size, num_epochs, model_file):
    indices = list(range(len(data)))
    np.random.shuffle(indices)
    split = int(np.floor(0.2 * len(data)))
    train_idx, valid_idx = indices[split:], indices[:split]

    # define samplers for obtaining training and validation batches
    train_sampler = SubsetRandomSampler(train_idx)
    valid_sampler = SubsetRandomSampler(valid_idx)

    # prepare data loaders
    train_loader = DataLoader(data, batch_size=batch_size, sampler=train_sampler)
    valid_loader = DataLoader(data, batch_size=batch_size, sampler=valid_sampler)

    valid_loss_min = np.Inf # set initial "min" to infinity

    for epoch in range(num_epochs):
        train_loss = 0.0
        valid_loss = 0.0

        model.train() # prep model for training
        for X_training, y_training in train_loader:
            X_training = X_training.to(cuda)
            y_training = y_training.to(cuda)
            # clear the gradients of all optimized variables
            optimizer.zero_grad()
            # forward pass: compute predicted outputs by passing inputs to the model
            X_sorted = X_training
            y_sorted = y_training

            loss = model.loss(model(X_sorted), y_sorted)
            # backward pass: compute gradient of the loss with respect to model parameters
            loss.backward()
            # perform a single optimization step (parameter update)
            optimizer.step()
            # update running training loss
            train_loss += loss.item() * X_training.size(0)

        model.eval() # prep model for evaluation
        for X_valid, y_valid in valid_loader:
            X_valid = X_valid.to(cuda)
            y_valid = y_valid.to(cuda)
            X_sorted = X_valid
            y_sorted = y_valid
            with torch.no_grad():
              loss = model.loss(model(X_sorted), y_sorted)
            # update running validation loss 
            valid_loss += loss.item() * X_valid.size(0)

        # print training/validation statistics 
        # calculate average loss over an epoch
        train_loss = train_loss/len(train_loader.dataset)
        valid_loss = valid_loss/len(valid_loader.dataset)

        print('Epoch: {} \tTraining Loss: {:.6f} \tValidation Loss: {:.6f}'.format(
            epoch + 1, 
            train_loss,
            valid_loss
            ))

        # save model if validation loss has decreased
        if valid_loss <= valid_loss_min:
            print('Validation loss decreased ({:.6f} --> {:.6f}).  Saving model ...'.format(
            valid_loss_min,
            valid_loss))
            save(model.state_dict(), model_file)
            valid_loss_min = valid_loss
        
        # decay learning rate
        # scheduler.step()
        
    return valid_loss_min

import torch.nn.functional as F
class BiLSTM(nn.Module):
    def __init__(self, output_dim, batch_size, embedding_dim, hidden_dim, linear_dim, drop_in, drop_out):
        super(BiLSTM, self).__init__()
        self.batch_size = batch_size
        self.hidden_dim = hidden_dim
        self.embedding_dim = embedding_dim
        self.linear_dim = linear_dim
        self.dropout_pct1 = drop_in
        self.dropout_pct2 = drop_out
        self.output_dim = output_dim
        self.loss_fn = nn.NLLLoss() 

        self.dropout_input = nn.Dropout(self.dropout_pct1)
        self.lstm = nn.LSTM(self.embedding_dim, self.hidden_dim, bidirectional=True, batch_first=True)
        self.dropout_output = nn.Dropout(self.dropout_pct2)
        self.linear = nn.Linear(self.hidden_dim * 2, self.linear_dim)
        self.elu = nn.ELU()
        self.out = nn.Linear(self.linear_dim, self.output_dim)
        
    def init_hidden(self, batch_size):
        hidden_a = torch.randn(2, batch_size, self.hidden_dim).to(cuda)
        hidden_b = torch.randn(2, batch_size, self.hidden_dim).to(cuda)

        return (torch.autograd.Variable(hidden_a), torch.autograd.Variable(hidden_b))

    def forward(self, X):
        hidden = self.init_hidden(X.shape[0])
        a, hidden = self.lstm(X, hidden)
        a = self.dropout_output(a[:,-1,])
        a = self.elu(self.linear(a)).reshape(a.shape[0], -1)
        a = self.out(a)
        return F.log_softmax(a)
    
    def loss(self, Y_hat, Y):
        return self.loss_fn(Y_hat, Y)

trained_emotion_model = BiLSTM(8, batch_size, len(trained_embedding_matrix[0]), hidden_dim, 
                                  linear_dim, drop_in, drop_out).to(cuda)
optimizer = SGD(trained_emotion_model.parameters(), lr=lr, momentum=momentum)
min_val_loss = trainAndEvalBiLSTM(trained_emotion_model, optimizer, trained_emotion_set, max_num_clauses, batch_size, num_epochs, 'blstm_trained.pt')

index = 3
print('Review:', data_trained['Review'][index])

input = np.array([data_trained['review_text_eae_embeddings'][index]])

print('Original Emotion Index:', data_trained['emotion'][index])

with torch.no_grad(): 
  prediction = trained_emotion_model(FloatTensor(input).to(cuda))[0]
  print('Predicted probabilities in log:', prediction)
  print('Predicted Emotion Index:', np.argmax(prediction.cpu().detach().numpy()))

"""## Training Cause Extraction Model"""

X = []
y = []

new_clause_eae_embeddings = []
for ind, row in train_df.iterrows():
  with torch.no_grad():
    tensor = FloatTensor(np.array([row['review_text_eae_embeddings']])).to(cuda)
  log_probabilities = trained_emotion_model(tensor).cpu().detach().numpy()[0]
  emotion_probabilities = np.exp(log_probabilities)
  clauses = row['clause_eae_embeddings']
  cind = 0
  cur_new_clause_eae_embeddings = []
  for cind in range(len(clauses)):
    curX = []
    for word_embedding in clauses[cind]:
      cur_embedding = []
      for emotion_probability in emotion_probabilities:
        cur_embedding.extend(emotion_probability * word_embedding)
      curX.append(cur_embedding)
    curX = np.array(curX)
    X.append(curX)
    
    cur_new_clause_eae_embeddings.append(curX)

    if row['cause'] == cind:
      y.append(1.0)
    else:
      y.append(0.0)
  new_clause_eae_embeddings.append(cur_new_clause_eae_embeddings)
train_df['new_clause_eae_embeddings'] = new_clause_eae_embeddings

class DFDatasetClauseExtraction(Dataset):
    def __init__(self, df):
        self.df = df

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        return FloatTensor(row["input"]), FloatTensor([row["label"]])

data = pd.DataFrame({'input' : X, 'label' : y})
trained_cause_set = DFDatasetClauseExtraction(data)

#model dimensions / parameters
hidden_dim = 1024
linear_dim = 80
drop_in = 0.2  # dropout layer percentages
drop_out = 0.5

# hyperparamters
batch_size = 1
num_epochs = 50
lr = 0.003
momentum = 0.9

import torch.nn.functional as F
class BiLSTMCauseExtraction(nn.Module):
  def __init__(self, output_dim, batch_size, embedding_dim, hidden_dim, linear_dim, drop_in, drop_out):
      super(BiLSTMCauseExtraction, self).__init__()
      self.batch_size = batch_size
      self.hidden_dim = hidden_dim
      self.embedding_dim = embedding_dim
      self.linear_dim = linear_dim
      self.dropout_pct1 = drop_in
      self.dropout_pct2 = drop_out
      self.output_dim = output_dim
      self.loss_fn = nn.BCELoss() #weight=Tensor(tag_weights))

      self.dropout_input = nn.Dropout(self.dropout_pct1)
      self.lstm = nn.LSTM(self.embedding_dim, self.hidden_dim, bidirectional=True, batch_first=True)
      self.dropout_output = nn.Dropout(self.dropout_pct2)
      self.linear = nn.Linear(self.hidden_dim * 2, self.linear_dim)
      self.elu = nn.ELU()
      self.out = nn.Linear(self.linear_dim, self.output_dim)
      
  def init_hidden(self, batch_size):
      # the weights are of the form (nb_layers, batch_size, nb_lstm_units)
      hidden_a = torch.randn(2, batch_size, self.hidden_dim).to(cuda)
      hidden_b = torch.randn(2, batch_size, self.hidden_dim).to(cuda)

      return (torch.autograd.Variable(hidden_a), torch.autograd.Variable(hidden_b))

  def forward(self, X):
      
      hidden = self.init_hidden(X.shape[0])
      
      a, hidden = self.lstm(X, hidden)
      a = self.dropout_output(a[:,-1,])
      a = self.elu(self.linear(a)).reshape(a.shape[0], -1)
      a = self.out(a)
      return F.sigmoid(a)
  
  def loss(self, Y_hat, Y):
      return self.loss_fn(Y_hat, Y)

def trainAndEvalBiLSTMCauseExtraction(model, optimizer, data, num_clauses, batch_size, num_epochs, model_file):
    indices = list(range(len(data)))
    np.random.shuffle(indices)
    split = int(np.floor(0.2 * len(data)))
    train_idx, valid_idx = indices[split:], indices[:split]

    # define samplers for obtaining training and validation batches
    train_sampler = SubsetRandomSampler(train_idx)
    valid_sampler = SubsetRandomSampler(valid_idx)

    # prepare data loaders
    train_loader = DataLoader(data, batch_size=batch_size, sampler=train_sampler)
    valid_loader = DataLoader(data, batch_size=batch_size, sampler=valid_sampler)

    valid_loss_min = np.Inf # set initial "min" to infinity

    for epoch in range(num_epochs):
        train_loss = 0.0
        valid_loss = 0.0

        model.train() # prep model for training
        for X_training, y_training in train_loader:
            X_training = X_training.to(cuda)
            y_training = y_training.to(cuda)
            # clear the gradients of all optimized variables
            optimizer.zero_grad()
            # forward pass: compute predicted outputs by passing inputs to the model
            X_sorted = X_training
            y_sorted = y_training

            loss = model.loss(model(X_sorted), y_sorted)
            # backward pass: compute gradient of the loss with respect to model parameters
            loss.backward()
            # perform a single optimization step (parameter update)
            optimizer.step()
            # update running training loss
            train_loss += loss.item() * X_training.size(0)

        model.eval() # prep model for evaluation
        for X_valid, y_valid in valid_loader:
            X_valid = X_valid.to(cuda)
            y_valid = y_valid.to(cuda)
            X_sorted = X_valid
            y_sorted = y_valid
            loss = model.loss(model(X_sorted), y_sorted)
            # update running validation loss 
            valid_loss += loss.item() * X_valid.size(0)

        # print training/validation statistics 
        # calculate average loss over an epoch
        train_loss = train_loss/len(train_loader.dataset)
        valid_loss = valid_loss/len(valid_loader.dataset)

        print('Epoch: {} \tTraining Loss: {:.6f} \tValidation Loss: {:.6f}'.format(
            epoch + 1, 
            train_loss,
            valid_loss
            ))

        # save model if validation loss has decreased
        if valid_loss <= valid_loss_min:
            print('Validation loss decreased ({:.6f} --> {:.6f}).  Saving model ...'.format(
            valid_loss_min,
            valid_loss))
            save(model.state_dict(), model_file)
            valid_loss_min = valid_loss
        
        # decay learning rate
        # scheduler.step()
        
    return valid_loss_min

trained_cause_model = BiLSTMCauseExtraction(1, batch_size, 8 * len(trained_embedding_matrix[0]), hidden_dim, 
                             linear_dim, drop_in, drop_out).to(cuda)
optimizer = SGD(trained_cause_model.parameters(), lr=lr, momentum=momentum)
min_val_loss = trainAndEvalBiLSTMCauseExtraction(trained_cause_model, optimizer, trained_cause_set, max_num_clauses, batch_size, num_epochs, 'blstm_trained_cause.pt')

index = 7
clause_embeddings = train_df['new_clause_eae_embeddings'][index]
clauses = train_df['clauses'][index]

print('Review:', data_trained['Review'][index])

best_clause_ind = -1
best_prob = -1
for cind in range(len(clauses)):
  print('Clause ' + str(cind) + ': ' + ' '.join(clauses[cind]))
  input = np.array(clause_embeddings[cind])

  with torch.no_grad(): 
    prediction = trained_cause_model(FloatTensor([input]).to(cuda))[0]
    print('Probability:', prediction.item())

    if prediction.item() > best_prob:
      best_prob = prediction.item()
      best_clause_ind = cind

print('Annotated Clause Index:', train_df['cause'][index])  
print('Predicted Clause Index:', best_clause_ind)

predicted_clause_indices = []
for index in range(train_df.shape[0]):
  clause_embeddings = train_df['new_clause_eae_embeddings'][index]
  clauses = train_df['clauses'][index]

  # print('Review:', data_trained['Review'][index])

  best_clause_ind = -1
  best_prob = -1
  for cind in range(len(clauses)):
    # print('Clause ' + str(cind) + ': ' + ' '.join(clauses[cind]))
    input = np.array(clause_embeddings[cind])

    with torch.no_grad(): 
      prediction = trained_cause_model(FloatTensor([input]).to(cuda))[0]
      # print('Probability:', prediction.item())

      if prediction.item() > best_prob:
        best_prob = prediction.item()
        best_clause_ind = cind
  predicted_clause_indices.append(best_clause_ind)
  # print('Annotated Clause Index:', train_df['cause'][index])  
  # print('Predicted Clause Index:', best_clause_ind)
train_df['Predicted Cause Index'] = predicted_clause_indices

train_df['clauses'] = train_df['clauses'].apply(lambda x : [' '.join(clause) for clause in x])

with open('predicted_dataset.pkl', 'wb') as file:
  pickle.dump(train_df, file)