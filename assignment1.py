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

# %%
ustoi = dict(df.groupby('userID')['userIDX'].unique().apply(lambda x: x[0]))
istoi = dict(df.groupby('itemID')['itemIDX'].unique().apply(lambda x: x[0]))

# %%
df.drop(labels=ft + ['hours_transformed', 'found_funny'], axis=1, inplace=True)
df.head()

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
df_played_train = df.iloc[:150000]
df_played_valid = df.iloc[150000:]

# %%
model = RankFM(factors=10,
               loss='warp',
               max_samples=300,
               learning_exponent=0.25,
               learning_schedule='invscaling')

# %%
# Construct a new validation set w/ negative pairs
neg_pairs = []
for review in df_played_valid.iterrows():
    review = review[1]
    sample = random.sample(itemset.difference(I[review['userID']]), k=1)[0]
    neg_pairs.append([review['userIDX'], istoi[sample]])
pos_pairs = df_played_valid[['userIDX', 'itemIDX']].to_numpy()
neg_pairs = np.array(neg_pairs)

def validate(model):
    pos_scores = model.predict(pos_pairs)
    neg_scores = model.predict(neg_pairs)
    acc = (np.mean(pos_scores >= 0) + np.mean(neg_scores < 0)) / 2
    print(f'Validation %: {acc * 100}')

# %%
for i in range(100):
    model.fit_partial(df_played_train[['userIDX', 'itemIDX']], item_features=text_embed, epochs=1, verbose=False)
    validate(model)

# %%
import pickle

# model_file = open('rankfm.obj', 'wb')
# pickle.dump(model, model_file)

# %% [markdown]
# #### Played dataset

# %%
test = pd.read_csv('./pairs_Played.csv')
test['itemID'] = test['gameID']
# Map unseen entries to default user (this user is already grouped with other users due to their few # of reviews in training set)
test['userID'] = test['userID'].map(lambda x: x if x in userset else 'u03473346')
test['userIDX'] = user_oe.transform(test[['userID']])
test['itemIDX'] = item_oe.transform(test[['gameID']])
test.drop(columns=['gameID', 'prediction'], inplace=True)

# %%
test.head()

# %%
scores = model.predict(test[['userIDX', 'itemIDX']])

# %%
testpred = pd.read_csv('./pairs_Played.csv')
testpred['prediction'] = (scores >= 0).astype(np.int32)

# %%
testpred.to_csv('./predictions_Played.csv', index=False)


