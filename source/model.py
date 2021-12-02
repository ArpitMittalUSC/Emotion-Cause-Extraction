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

cuda = torch.device('cuda:0')

seed = 39542
np.random.seed(seed)
random.seed(seed)
torch.manual_seed(seed)

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
        return FloatTensor(row["clauses"]), Tensor(row["label"])
    
class BiLSTM(nn.Module):
    
    def __init__(self, num_clauses, output_dim, batch_size, embedding_dim, hidden_dim, linear_dim, drop_in, drop_out):
        super(BiLSTM, self).__init__()
        self.batch_size = batch_size
        self.num_clauses = num_clauses
        self.hidden_dim = hidden_dim
        self.embedding_dim = embedding_dim
        self.linear_dim = linear_dim
        self.dropout_pct1 = drop_in
        self.dropout_pct2 = drop_out
        self.output_dim = output_dim
        self.loss_fn = nn.CrossEntropyLoss() #weight=Tensor(tag_weights))

        self.dropout_input = nn.Dropout(self.dropout_pct1)
        self.lstm = nn.LSTM(self.embedding_dim, self.hidden_dim, bidirectional=True, batch_first=True)
        self.dropout_output = nn.Dropout(self.dropout_pct2)
        self.linear = nn.Linear(self.hidden_dim * 2, self.linear_dim)
        self.elu = nn.ELU()
        self.out = nn.Linear(self.linear_dim * self.num_clauses, self.output_dim)
        
    def init_hidden(self, batch_size):
        # the weights are of the form (nb_layers, batch_size, nb_lstm_units)
        hidden_a = torch.randn(2, batch_size, self.hidden_dim).to(cuda)
        hidden_b = torch.randn(2, batch_size, self.hidden_dim).to(cuda)

        return (torch.autograd.Variable(hidden_a), torch.autograd.Variable(hidden_b))

    def forward(self, X, X_lengths):
        
        self.hidden = self.init_hidden(X.shape[0])
        
        X = self.dropout_input(X)
        # pack sequence
        # print(X.shape)
        packed_input = pack_padded_sequence(X, X_lengths, batch_first=True)
        
        a, self.hidden = self.lstm(packed_input, self.hidden)
        
        a, _ = pad_packed_sequence(a, batch_first=True, total_length=self.num_clauses)
        a = self.dropout_output(a)
        # print(a.shape)
        
        # a = a.contiguous().view(-1, a.shape[-2])
        a = self.elu(self.linear(a)).reshape(a.shape[0], -1)
        a = self.out(a)
        return a
    
    def loss(self, Y_hat, Y):
        #print(Y_hat.shape)
        #print(Y.shape)
        OH_Y = FloatTensor(np.eye(self.output_dim)[Y.cpu()]).to(cuda)
        #print(OH_Y).shape
        return self.loss_fn(Y_hat, Y)
    
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
            # sort inputs and labels first
            X_lengths = torch.stack([Tensor(num_clauses - np.argmin([(clause == 0).all() for clause in torch.flip(x, (0,)) ])) for x in X_training.cpu()]).sort(descending=True)
            X_sorted = X_training[X_lengths.indices]
            y_sorted = y_training[X_lengths.indices]
            loss = model.loss(model(X_sorted, X_lengths.values), y_sorted)
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
            X_lengths = torch.stack([Tensor(num_clauses - np.argmin([(clause == 0).all() for clause in torch.flip(x, (0,)) ])) for x in X_valid.cpu()]).sort(descending=True)
            X_sorted = X_valid[X_lengths.indices]
            y_sorted = y_valid[X_lengths.indices]
            loss = model.loss(model(X_sorted, X_lengths.values), y_sorted)
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


def main():
    tokenize = TweetTokenizer().tokenize

    ## Need to know the name of output file here
    clause_extracted_path = '../outputs/dataset.pkl'

    with open(clause_extracted_path, 'rb') as file:
        raw_data = pickle.load(file)
        train_df = pd.DataFrame(raw_data)[["Clauses", "emotion", "causeClauseIndex"]].rename(columns={'Clauses': 'clauses', 
                                                                                                    'causeClauseIndex': 'cause'})

    max_num_clauses = max(len(doc) for doc in train_df["clauses"])
    num_labels = 8
    max_word_size = 20   # this is passed into TrimWords to avoid super long words

    train_df.loc[:, "clauses"] = train_df["clauses"].apply(lambda x: tokenize_clauses(x, tokenize, max_word_size))
    vocab = get_vocab(train_df["clauses"])
    max_num_words = max([[len(clause) for clause in doc] for doc in train_df["clauses"]])[0]
    train_df.loc[:, "clauses"] = train_df["clauses"].apply(lambda x: encode_words(x, vocab, max_num_words))
    train_df.loc[:, "emotion"] = train_df["emotion"].apply(int)
    train_df.loc[:, "cause"] = train_df["cause"].apply(int)
    pad_with_clauses(train_df["clauses"], max_num_words, max_num_clauses)

    train_df = train_df.drop(np.where(np.array(0) == [np.argmin(x) for x in train_df["clauses"]])[0])

    # get word2vec embeddings to use as a control
    with open('../outputs/wordEmbeddings.pkl', 'rb') as file:
        pretrained_wv = pickle.load(file)

    # get our emotion-aware embeddings
    with open('../outputs/emotionAwareEmbeddings.pkl', 'rb') as file:
        trained_wv = pickle.load(file)

    pretrained_embedding_matrix = unpack_embeddings(pretrained_wv, 300, vocab)
    trained_embedding_matrix = unpack_embeddings(trained_wv, 300, vocab, pretrained_wv)
    
    data_pretrained = train_df
    data_trained = train_df.copy()
    data_pretrained.loc[:, "clauses"] = data_pretrained["clauses"].apply(lambda doc: doc_to_embeddings(doc, pretrained_embedding_matrix))
    data_trained.loc[:, "clauses"] = data_trained["clauses"].apply(lambda doc: doc_to_embeddings(doc, trained_embedding_matrix))
    pretrained_emotion_set = DFDataset(data_pretrained[["clauses", "emotion"]].rename(columns={'emotion': 'label'}))
    trained_emotion_set = DFDataset(data_trained[["clauses", "emotion"]].rename(columns={'emotion': 'label'}))
    
    #model dimensions / parameters
    hidden_dim = 256
    linear_dim = 80
    drop_in = 0.2  # dropout layer percentages
    drop_out = 0.3

    # hyperparamters
    batch_size = 2
    num_epochs = 300
    lr = 0.002
    momentum = 0.9

    pretrained_emotion_model = BiLSTM(max_num_clauses, 8, batch_size, len(pretrained_embedding_matrix[0]), hidden_dim, 
                                     linear_dim, drop_in, drop_out).to(cuda)
    optimizer = SGD(pretrained_emotion_model.parameters(), lr=lr, momentum=momentum)
    min_val_loss = trainAndEvalBiLSTM(pretrained_emotion_model, optimizer, pretrained_emotion_set, max_num_clauses, batch_size, num_epochs, 'blstm_pretrained.pt')

    trained_emotion_model = BiLSTM(max_num_clauses, 8, batch_size, len(trained_embedding_matrix[0]), hidden_dim, 
                                    linear_dim, drop_in, drop_out).to(cuda)
    optimizer = SGD(trained_emotion_model.parameters(), lr=lr, momentum=momentum)
    min_val_loss = trainAndEvalBiLSTM(trained_emotion_model, optimizer, trained_emotion_set, max_num_clauses, batch_size, num_epochs, 'blstm_trained.pt')

    pretrained_emotion_model.eval()
    trained_emotion_model.eval()

    clause_data_pretrained, clause_data_trained = data_pretrained.copy(), data_trained.copy()

    softmax = nn.Softmax()
    
    Y_hat = []
    for i, (X, y) in enumerate(DataLoader(pretrained_emotion_set, batch_size=8)):
        X = X.to(cuda)
        X_lengths = torch.stack([Tensor(max_num_clauses - np.argmin([(clause == 0).all() for clause in torch.flip(x, (0,)) ])) for x in X.cpu()]).sort(descending=True)
        X_sorted = X[X_lengths.indices]

        y_predict = pretrained_emotion_model(X_sorted, X_lengths.values).data[X_lengths.indices.argsort(0)].cpu().view(-1, 8)
        Y_hat.extend( softmax(y_predict).tolist() )

    new_col = []
    for i in range(len(clause_data_pretrained)):
        doc, emotions = clause_data_pretrained["clauses"].iloc[i], Y_hat[i]
        new_doc = np.zeros((doc.shape[0], doc.shape[1] * num_labels))
        for j in range(len(doc)):
            new_doc[j] = np.concatenate([doc[j] * prob for prob in Y_hat[i]])
        new_col.append(new_doc)
    clause_data_pretrained.loc[:, "clauses"] = new_col

    Y_hat = []
    for i, (X, y) in enumerate(DataLoader(trained_emotion_set, batch_size=8)):
        X = X.to(cuda)
        X_lengths = torch.stack([Tensor(max_num_clauses - np.argmin([(clause == 0).all() for clause in torch.flip(x, (0,)) ])) for x in X.cpu()]).sort(descending=True)
        X_sorted = X[X_lengths.indices]

        y_predict = trained_emotion_model(X_sorted, X_lengths.values).data[X_lengths.indices.argsort(0)].cpu().view(-1, 8)
        Y_hat.extend( softmax(y_predict).tolist() )
    
    new_col = []
    for i in range(len(clause_data_trained)):
        doc, emotions = clause_data_trained["clauses"].iloc[i], Y_hat[i]
        new_doc = np.zeros((doc.shape[0], doc.shape[1] * num_labels))
        for j in range(len(doc)):
            new_doc[j] = np.concatenate([doc[j] * prob for prob in Y_hat[i]])
        new_col.append(new_doc)
    clause_data_trained.loc[:, "clauses"] = new_col

    pretrained_cause_set = DFDataset(clause_data_pretrained[["clauses", "cause"]].rename(columns={'cause': 'label'}))
    trained_cause_set = DFDataset(clause_data_trained[["clauses", "cause"]].rename(columns={'cause': 'label'}))

    #model dimensions / parameters
    hidden_dim = 1024
    linear_dim = 80
    drop_in = 0.2  # dropout layer percentages
    drop_out = 0.6

    # hyperparamters
    batch_size = 2
    num_epochs = 500
    lr = 0.001
    momentum = 0.9

    pretrained_cause_model = BiLSTM(max_num_clauses, max_num_clauses, batch_size, 8 * len(pretrained_embedding_matrix[0]), 
                                    hidden_dim, linear_dim, drop_in, drop_out).to(cuda)
    optimizer = SGD(pretrained_cause_model.parameters(), lr=lr, momentum=momentum)
    min_val_loss = trainAndEvalBiLSTM(pretrained_cause_model, optimizer, pretrained_cause_set, max_num_clauses, batch_size, num_epochs, 'blstm_pretrained_cause.pt')

    trained_cause_model = BiLSTM(max_num_clauses, max_num_clauses, batch_size, 8 * len(trained_embedding_matrix[0]), hidden_dim, 
                                linear_dim, drop_in, drop_out).to(cuda)
    optimizer = SGD(trained_cause_model.parameters(), lr=lr, momentum=momentum)
    min_val_loss = trainAndEvalBiLSTM(trained_cause_model, optimizer, trained_cause_set, max_num_clauses, batch_size, num_epochs, 'blstm_trained_cause.pt')

    pretrained_cause_model.eval()
    trained_cause_model.eval()

    Y_hat = []
    for i, (X, y) in enumerate(DataLoader(pretrained_cause_set, batch_size=8)):
        X = X.to(cuda)
        X_lengths = torch.stack([Tensor(max_num_clauses - np.argmin([(clause == 0).all() for clause in torch.flip(x, (0,)) ])) for x in X.cpu()]).sort(descending=True)
        X_sorted = X[X_lengths.indices]

        y_predict = pretrained_cause_model(X_sorted, X_lengths.values).data[X_lengths.indices.argsort(0)].cpu().view(-1, 8)
        Y_hat.extend( softmax(y_predict).tolist() )
    
    pretrained_cause_predictions = np.argmax(Y_hat, axis=1)

    Y_hat = []
    for i, (X, y) in enumerate(DataLoader(trained_cause_set, batch_size=8)):
        X = X.to(cuda)
        X_lengths = torch.stack([Tensor(max_num_clauses - np.argmin([(clause == 0).all() for clause in torch.flip(x, (0,)) ])) for x in X.cpu()]).sort(descending=True)
        X_sorted = X[X_lengths.indices]

        y_predict = trained_cause_model(X_sorted, X_lengths.values).data[X_lengths.indices.argsort(0)].cpu().view(-1, 8)
        Y_hat.extend( softmax(y_predict).tolist() )
    
    trained_cause_predictions = np.argmax(Y_hat, axis=1)
    pred_cause_embeddings = np.array([data_trained["clauses"].iloc[i][trained_cause_predictions[i]] for i in range(len(data_trained))])

    output_df = pd.DataFrame({'cause': train_df['cause'], 'embedding': list(pred_cause_embeddings)})
    output_df.to_pickle("../outputs/modelOutput.pkl")

if __name__ == "__main__":
    main()