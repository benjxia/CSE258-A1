# %% [markdown]
# # CSE 258: Assignment 1
# ### Benjamin Xia

# %% [markdown]
# ### Setup

# %%
import numpy as np
import pandas as pd

from sklearn import preprocessing
from sklearn.decomposition import LatentDirichletAllocation
from sklearn import feature_extraction

from rankfm.rankfm import RankFM

import random
from collections import defaultdict
from tqdm import tqdm
import gzip

import os
import pickle
import copy

RANDOM_SEED = 0
random.seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)

import warnings
warnings.filterwarnings("ignore", category=UserWarning)

# %% [markdown]
# ### Preprocessing

# %% [markdown]
# #### Preprocess user/item ID's, compensation, early_access, and time

# %%
user_oe = preprocessing.OrdinalEncoder(dtype=np.int32, min_frequency=5, handle_unknown='use_encoded_value', unknown_value=6710)
item_oe = preprocessing.OrdinalEncoder(dtype=np.int32, min_frequency=5)

itemset = set() # Set of all unique users
userset = set() # Set of all unique items
U = defaultdict(set)
I = defaultdict(set)

ft = ['early_access', 'compensation'] # features unavailable/cannot be approximated in inference
def read_json(path):
    f: gzip.GzipFile = gzip.open(path)
    f.readline()
    for line in f:
        entry = eval(line)
        yield entry

# Encode userID and itemID as integers
def process_data():
    global itemset, userset, U, I
    data = []
    for entry in read_json('train.json.gz'):
        data.append(entry)

    df: pd.DataFrame = pd.DataFrame(data)
    del data
    itemset = set(df['gameID'].unique())
    userset = set(df['userID'].unique())

    U = dict(df.groupby('gameID')['userID'].unique())
    I = dict(df.groupby('userID')['gameID'].unique())
    U = { g : set(U[g]) for g in U }
    I = { u : set(I[u]) for u in I }

    df['userIDX'] = user_oe.fit_transform(df[['userID']])
    df['itemIDX'] = item_oe.fit_transform(df[['gameID']])
    df.rename({'gameID' : 'itemID'}, axis=1, inplace=True)


    df.drop(labels=['hours', 'user_id', 'date'], axis=1, inplace=True)


    # Get features that won't be available
    df.fillna(value=0, axis=1, inplace=True)
    df['compensation'] = df['compensation'].map(lambda x : x if x == 0 else 1)
    df[['early_access', 'compensation']] = df[['early_access', 'compensation']].astype(np.int32)

    time_label = df['hours_transformed']

    return df, time_label

df, time_label = process_data()
user_mean = df.groupby('userIDX')[ft].mean()
item_mean = df.groupby('itemIDX')[ft].mean()
df.drop(labels=ft + ['hours_transformed', 'found_funny'], axis=1, inplace=True)

# %%
ustoi = dict(df.groupby('userID')['userIDX'].unique().apply(lambda x: x[0]))
istoi = dict(df.groupby('itemID')['itemIDX'].unique().apply(lambda x: x[0]))

# %% [markdown]
# #### Preprocess user text and convert to descriptors

# %%
def get_text_embedding():
    if not os.path.isfile('./text_embed.npy'): # Generate new descriptors for each review using pretrained transformer
        dftext = df.groupby('itemIDX')['text'].apply(' '.join).reset_index()
        counter = feature_extraction.text.CountVectorizer(min_df=0.05, max_df=0.5, stop_words='english', max_features=2000, ngram_range=(1, 2))
        wordcount = counter.fit_transform(dftext['text'])
        LDA = LatentDirichletAllocation(n_components=20, random_state=RANDOM_SEED)
        text_embed = LDA.fit_transform(wordcount)
        np.save('text_embed.npy', text_embed)
    else: # Text descriptors already computed
        text_embed = np.load('./text_embed.npy')

    return text_embed

text_embed = get_text_embedding()
text_embed = text_embed / np.linalg.norm(text_embed, axis=1)[...,None]

df.drop('text', axis=1, inplace=True)


# %%
text_embed = np.concatenate((np.arange(0, len(text_embed))[:,  None], text_embed, item_mean.to_numpy()), axis=1)

# %%
df_train = df.iloc[:150000]
df_time_train_label = time_label[:150000]
df_valid = df.iloc[150000:]
df_time_valid_label = time_label[150000:]

# %% [markdown]
# ### Played Predictions

# %%
played_model = RankFM(factors=10,
               loss='warp',
               max_samples=300,
               learning_exponent=0.25,
               learning_schedule='invscaling')

# %%
# Construct a new validation set w/ negative pairs
neg_pairs = []
for review in df_valid.iterrows():
    review = review[1]
    sample = random.sample(itemset.difference(I[review['userID']]), k=1)[0]
    neg_pairs.append([review['userIDX'], istoi[sample]])
pos_pairs = df_valid[['userIDX', 'itemIDX']].to_numpy()
neg_pairs = np.array(neg_pairs)

def played_validate(model):
    pos_scores = model.predict(pos_pairs)
    neg_scores = model.predict(neg_pairs)
    acc = (np.mean(pos_scores >= 0) + np.mean(neg_scores < 0)) / 2
    print(f'Validation %: {acc * 100}')
    return acc

# %%
best_model = None
best_acc = 0
for i in range(50):
    played_model.fit_partial(df_train[['userIDX', 'itemIDX']], item_features=text_embed, epochs=4, verbose=False)
    acc = played_validate(played_model)
    if acc > best_acc:
        best_model = copy.deepcopy(played_model)
        best_acc = acc

model_file = open('rankfm.obj', 'wb')
pickle.dump(best_model, model_file)

# %% [markdown]
# #### Make and write predictions

# %%
test = pd.read_csv('./pairs_Played.csv')
testpred = test.copy()
test['itemID'] = test['gameID']
# Map unseen entries to default user (this user is already grouped with other users due to their few # of reviews in training set)
test['userID'] = test['userID'].map(lambda x: x if x in userset else 'u03473346')
test['userIDX'] = user_oe.transform(test[['userID']])
test['itemIDX'] = item_oe.transform(test[['gameID']])
test.drop(columns=['gameID', 'prediction'], inplace=True)
scores = best_model.predict(test[['userIDX', 'itemIDX']])
testpred = pd.read_csv('./pairs_Played.csv')
testpred['prediction'] = (scores >= 0).astype(np.int32)
testpred.to_csv('./predictions_Played.csv', index=False)

# %% [markdown]
# ### Time Prediction

# %%
import torch
from torch import nn
from torch import optim
from torch.utils.data import Dataset, DataLoader

# %%
torch.set_default_tensor_type(torch.DoubleTensor)
device = 'cuda' if torch.cuda.is_available() else 'cpu'
class TimeDataset(Dataset):
    def __init__(self, df, label) -> None:
        super().__init__()
        self.df = df
        self.label = label
    def __len__(self):
        return len(self.df)
    def __getitem__(self, index):
        row = self.df.iloc[index]
        itemIDX = row['itemIDX']

        # Build positive pair
        data = np.concatenate((row[2:].to_numpy().astype(np.float32),
                              text_embed[itemIDX][1:].astype(np.float32)))
        label = self.label[index]
        return torch.from_numpy(data).to(dtype=torch.float64), torch.tensor(label)

# %%
class FactorizationMachine(nn.Module):
    def __init__(self, n_user, n_item, n_feature, latent_dim, weight=True) -> None:
        """
        n_user: Number of unique users
        n_item: Number of unique items
        n_feature: Number of extra features to use
        latent_dim: Dimension of latent representations of users/items/features
        """
        super().__init__()
        self.n_user = n_user
        self.n_item = n_item
        self.n_feature = n_feature
        self.latent_dim = latent_dim
        self.weight = weight
        self.user_latent = nn.Embedding(n_user, latent_dim)
        self.item_latent = nn.Embedding(n_item, latent_dim)
        if self.n_feature > 0:
            self.feat_latent = nn.Parameter(torch.randn(n_feature, latent_dim), requires_grad=True)
            self.feat_weight = nn.Linear(n_feature, 1)
        if self.weight:
            self.user_weight = nn.Embedding(n_user, 1)
            self.item_weight = nn.Embedding(n_item, 1)
        # "alpha" or "w_0" term will be absorbed into feat_weight linear's bias
    def forward(self, x) -> torch.Tensor:
        """
        Input shape: batch_size x (user idx, item idx, features) - 2 dimensional
        Returns: n x 1 tensor of predictions
        """
        if len(x.size()) == 1:
            x = x.unsqueeze(0)
        # f(u, i) = w_0 + \sum_{j=1}^{d} w_j * x_j
        out = torch.zeros((x.size()[0], 1), device=device)
        if self.n_feature > 0 and self.weight:
            out += self.feat_weight(x[:, 2:])
        users = x[:, 0].to(dtype=torch.int32)
        items = x[:, 1].to(dtype=torch.int32)
        if self.weight:
            out += self.user_weight(users)
            out += self.item_weight(items)
        # Nested summation thingy
        # Interactions between users/items and features
        u_embed = self.user_latent(users)
        i_embed = self.item_latent(items)
        out += (u_embed * i_embed).sum(dim=1).unsqueeze(-1)   # Dot product between user and item latent representations
        if self.n_feature > 0:
            # Interactions between features
            xfeature = x[:, 2:]
            out += ((u_embed @ self.feat_latent.T) * xfeature).sum(dim=1).unsqueeze(-1) # Dot product between user and feature latent representations

        return out


# %%
time_model = FactorizationMachine(len(df['userID'].unique()), len(df['itemID'].unique()), 22, 10, True).to(device) # 24 features

# %%
batch_sz=20
print_iter=1000
time_train_ds = TimeDataset(df_train, df_time_train_label)
time_train_dl = DataLoader(dataset=time_train_ds,
                       batch_size=batch_sz,
                       shuffle=True, num_workers=2)
time_valid_ds = TimeDataset(df_valid.reset_index(drop=True), df_time_valid_label.reset_index(drop=True))
time_valid_dl = DataLoader(dataset=time_valid_ds,
                             batch_size=1,
                             num_workers=2)
optimizer = optim.SGD(time_model.parameters(), lr=0.01, momentum=0.9)
criterion = nn.MSELoss()

# %%
time_train_ds[0][0].size()

# %%
def time_validate(model):
    with torch.no_grad():
        loss = 0
        for i in range(len(time_valid_ds)):
            data, label = time_valid_ds[i]
            pred = model(data.to(device))
            loss += criterion(label, pred)
        return torch.sqrt(loss / len(time_valid_ds)).item()

# %%
for i in range(10):
    running_loss = 0
    for i, (data, label) in tqdm(enumerate(time_train_dl)):
        preds = time_model(data.to(device))
        loss = criterion(label, preds)
        running_loss += loss
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        if i % 1000 == 0:
            running_loss /= 1000
            print(f'RMSE: {torch.sqrt(running_loss).item()}')
            running_loss = 0
    print(f'Validation RMSE: {time_validate(time_model)}')

# %% [markdown]
# #### Make and write predictions

# %%
test = pd.read_csv('./pairs_Hours.csv')
testpred = test.copy()
test['itemID'] = test['gameID']
# Map unseen entries to default user (this user is already grouped with other users due to their few # of reviews in training set)
test['userID'] = test['userID'].map(lambda x: x if x in userset else 'u03473346')
test['userIDX'] = user_oe.transform(test[['userID']])
test['itemIDX'] = item_oe.transform(test[['gameID']])
test.drop(columns=['gameID', 'prediction'], inplace=True)

preds = []
for i in range(len(test)):
    row = test.iloc[i]
    itemIDX = row['itemIDX']
    # Build positive pair
    data = np.concatenate((row[2:].to_numpy().astype(np.float64),
                            text_embed[itemIDX][1:].astype(np.float64)))
    preds.append(time_model(torch.tensor(data).to(device)).item())


testpred = pd.read_csv('./pairs_Hours.csv')
testpred['prediction'] = preds
testpred.to_csv('./predictions_Hours.csv', index=False)


