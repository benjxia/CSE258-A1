# %% [markdown]
# # CSE 258: Assignment 1
# ### Benjamin Xia

# %% [markdown]
# ### Setup

# %%
import numpy as np
import pandas as pd
from scipy import sparse

from sklearn import preprocessing, feature_extraction, linear_model
from sklearn.decomposition import LatentDirichletAllocation
from sklearn.model_selection import KFold, GridSearchCV

from rankfm.rankfm import RankFM
from fastFM import als, sgd

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

valid = False

# %% [markdown]
# ### Preprocessing

# %% [markdown]
# #### Preprocess user/item ID's, compensation, early_access, and time

# %%
user_oe = preprocessing.OrdinalEncoder(dtype=np.int32, min_frequency=5, handle_unknown='error')
item_oe = preprocessing.OrdinalEncoder(dtype=np.int32)

itemset = set() # Set of all unique users
userset = set() # Set of all unique items

ft = ['early_access', 'compensation'] # features unavailable/cannot be approximated in inference

def read_json(path):
    f: gzip.GzipFile = gzip.open(path)
    f.readline()
    for line in f:
        entry = eval(line)
        yield entry

# Encode userID and itemID as integers
def process_data():
    data = []
    data = []
    for entry in read_json('train.json.gz'):
        data.append(entry)
    df: pd.DataFrame = pd.DataFrame(data)
    del data

    df['userIDX'] = user_oe.fit_transform(df[['userID']])
    df['itemIDX'] = item_oe.fit_transform(df[['gameID']])
    df.rename({'gameID' : 'itemID'}, axis=1, inplace=True)

    df.drop(labels=['hours', 'user_id', 'date', 'userID', 'itemID'], axis=1, inplace=True)

    # Get features that won't be available
    df.fillna(value=0, axis=1, inplace=True)
    df['compensation'] = df['compensation'].map(lambda x : x if x == 0 else 1)
    df[['early_access', 'compensation']] = df[['early_access', 'compensation']].astype(np.int32)

    time_label = df['hours_transformed'].to_numpy()
    user_mean_ft = df.groupby('userIDX')[ft].mean()
    item_mean_ft = df.groupby('itemIDX')[ft].mean()
    df.drop(labels=ft + ['hours_transformed', 'found_funny'], axis=1, inplace=True)
    return df, time_label, user_mean_ft.to_numpy(), item_mean_ft.to_numpy()

df, time_label, user_mean_ft, item_mean_ft = process_data()

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
pairs = df[['userIDX', 'itemIDX']].to_numpy(dtype=np.int32)
item_ft = np.concatenate((np.arange(0, len(text_embed))[:,  None], item_mean_ft, text_embed, df['itemIDX'].value_counts().sort_index().to_numpy()[:, None] / np.max(df['itemIDX'].value_counts())), axis=1)

U = dict(df.groupby('itemIDX')['userIDX'].unique())
I = dict(df.groupby('userIDX')['itemIDX'].unique())
U = { g : set(U[g]) for g in U }
I = { u : set(I[u]) for u in I }
itemset = set(df['itemIDX'].unique())
userset = set(df['userIDX'].unique())

# %% [markdown]
# ### Played Predictions

# %%
# Construct a new validation set w/ negative pairs
def gen_neg_samples(pairs):
    neg_pairs = np.zeros_like(pairs)
    neg_pairs[:, 0] = pairs[:, 0]

    for i in range(len(pairs)):
        sample = random.sample(itemset.difference(I[pairs[i, 0]]), k=1)[0]
        neg_pairs[i, 1] = sample

    return neg_pairs

# %% [markdown]
# #### Played Model Selection

# %%
kf = KFold(n_splits=20, shuffle=True, random_state=RANDOM_SEED)
accs = []
for split, (train, test) in enumerate(kf.split(pairs)):
    popularity = df.iloc[train]['itemIDX'].value_counts().sort_index().to_numpy()
    # Generate training pairs for fold
    pos_train_pairs = pairs[train]
    neg_train_pairs = gen_neg_samples(pos_train_pairs)
    pos_valid_pairs = pairs[test]
    neg_valid_pairs = gen_neg_samples(pos_valid_pairs)
    # Train Models
    played_model = RankFM(factors=5,
                loss='bpr',
                max_samples=300,
                beta=1.0,
                learning_schedule='invscaling')
    played_model.fit(pairs[train], item_features=item_ft, epochs=200)
    pos_scores = played_model.predict(pos_train_pairs)
    neg_scores = played_model.predict(neg_train_pairs)
    pos_ft = popularity[pos_train_pairs[:, 1, None]]
    neg_ft = popularity[neg_train_pairs[:, 1, None]]
    pos_ft = np.column_stack((pos_scores, pos_ft))
    neg_ft = np.column_stack((neg_scores, neg_ft))
    clf = linear_model.LogisticRegression()
    ft = np.concatenate((pos_ft, neg_ft), axis=0)
    label = np.concatenate((np.ones(len(pos_ft)), np.zeros(len(pos_ft))))
    clf.fit(ft, label)

    # Validation
    pos_scores = played_model.predict(pos_valid_pairs)
    neg_scores = played_model.predict(neg_valid_pairs)
    pos_ft = popularity[pos_valid_pairs[:, 1, None]]
    neg_ft = popularity[neg_valid_pairs[:, 1, None]]
    pos_ft = np.column_stack((pos_scores, pos_ft))
    neg_ft = np.column_stack((neg_scores, neg_ft))
    ft = np.concatenate((pos_ft, neg_ft), axis=0)
    label = np.concatenate((np.ones(len(pos_ft)), np.zeros(len(pos_ft))))
    acc = clf.score(ft, label)
    accs.append(acc)
    print(f'Fold {split + 1}: {acc * 100}%')

print(f'Overall: {np.mean(accs) * 100}%')

# %%
logistic = False
kf = KFold(n_splits=20, shuffle=True)
accs = []
for split, (train, valid) in enumerate(kf.split(pairs)):
    popularity = df.iloc[train]['itemIDX'].value_counts().sort_index().to_numpy()
    # Generate training pairs for fold
    pos_train_pairs = pairs[train]
    neg_train_pairs = gen_neg_samples(pos_train_pairs)
    pos_valid_pairs = pairs[valid]
    neg_valid_pairs = gen_neg_samples(pos_valid_pairs)

    # Fit models
    played_model = RankFM(factors=4,
                loss='warp',
                max_samples=300,
                beta=1.0,
                learning_schedule='invscaling')

    played_model.fit(pos_train_pairs, item_features=item_ft, epochs=200)
    if logistic:
        pos_scores = played_model.predict(pos_train_pairs)
        neg_scores = played_model.predict(neg_train_pairs)
        pos_ft = np.column_stack((pos_scores, popularity[pos_train_pairs[:, 1]]))
        neg_ft = np.column_stack((neg_scores, popularity[neg_train_pairs[:, 1]]))
        clf = linear_model.LogisticRegression()
        ft = np.concatenate((pos_ft, neg_ft), axis=0)
        label = np.concatenate((np.ones(len(pos_ft)), np.zeros(len(pos_ft))))
        clf.fit(ft, label)

    # Validate
    pos_scores = played_model.predict(pos_valid_pairs)
    neg_scores = played_model.predict(neg_valid_pairs)
    pos_scores[np.isnan(pos_scores)] = 0
    neg_scores[np.isnan(neg_scores)] = 0
    if logistic:
        pos_ft = np.column_stack((pos_scores, popularity[pos_valid_pairs[:, 1]]))
        neg_ft = np.column_stack((neg_scores, popularity[neg_valid_pairs[:, 1]]))
        ft = np.concatenate((pos_ft, neg_ft), axis=0)
        label = np.concatenate((np.ones(len(pos_ft)), np.zeros(len(pos_ft))))
        acc = clf.score(ft, label)
    else:
        median = np.median(np.concatenate((pos_scores, neg_scores)))
        acc = ((np.mean(pos_scores >= median)) + (np.mean(neg_scores < median))) / 2

    accs.append(acc)
    print(f'Fold {split + 1}: {acc * 100}%')

print(f'Overall: {np.mean(accs) * 100}%')

# %%
played_model = RankFM(factors=4,
                loss='warp',
                max_samples=300,
                beta=1.0,
                learning_schedule='invscaling')
popularity = df['itemIDX'].value_counts().sort_index().to_numpy()
played_model.fit(pairs, item_features=item_ft, epochs=200)
pos_scores = played_model.predict(pairs)
neg_pairs = gen_neg_samples(pairs)
neg_scores = played_model.predict(neg_pairs)
pos_ft = np.column_stack((pos_scores, popularity[pairs[:, 1]]))
neg_ft = np.column_stack((neg_scores, popularity[neg_pairs[:, 1]]))
if logistic:
    clf = linear_model.LogisticRegression()
    ft = np.concatenate((pos_ft, neg_ft), axis=0)
    label = np.concatenate((np.ones(len(pos_ft)), np.zeros(len(pos_ft))))
    clf.fit(ft, label)

# %% [markdown]
# #### Make and write predictions

# %%
test_df = pd.read_csv('./pairs_Played.csv')
testpred = test_df.copy()
# # Map unseen entries to default user (this user is already grouped with other users due to their few # of reviews in training set)
test_df['userID'] = test_df['userID'].map(lambda x: x if x in user_oe.categories_[0] else 'u03473346')
test_df['itemID'] = test_df['gameID']
test_df['userIDX'] = user_oe.transform(test_df[['userID']])
test_df['itemIDX'] = item_oe.transform(test_df[['gameID']])
test_df.drop(columns=['gameID', 'prediction'], inplace=True)
test_pairs = test_df[['userIDX', 'itemIDX']].to_numpy()
scores = played_model.predict(test_pairs)
if logistic:
    test_pop = popularity[test_pairs[:, 1]]
    ft = np.column_stack((scores, test_pop))
    scores = clf.predict_log_proba(ft)[:, 1]
testpred = pd.read_csv('./pairs_Played.csv')
testpred['prediction'] = scores
medians = testpred.groupby('userID')['prediction'].median()
preds = []
for i, row in testpred.iterrows():
    if scores[i] >= medians[row['userID']]:
        preds.append(1)
    else:
        preds.append(0)
testpred['prediction'] = preds
testpred.to_csv('./predictions_Played.csv', index=False)

# %% [markdown]
# ### Time Prediction

# %% [markdown]
# #### FastFM,(this sucks but not as much, with or without features)

# %%
def convert_sparse_df(df: pd.DataFrame, feat=True):
    if feat == True:
        datum = sparse.lil_matrix((len(df), len(userset) + len(itemset) + 22))
    else:
        datum = sparse.lil_matrix((len(df), len(userset) + len(itemset)))
    for i, (idx, row) in enumerate(df.iterrows()):
        user = row['userIDX']
        item = row['itemIDX']
        datum[i, user] = 1
        datum[i, len(userset) + item] = 1
        if feat:
            datum[i, len(userset) + len(itemset):] = text_embed[item, 1:]
    return datum

all_sparse = convert_sparse_df(df, False)
all_sparse_ft = convert_sparse_df(df, True)



# %% [markdown]
# #### Time Model Model Selection

# %%
# param_grid = {
#     'n_iter': range(9, 13),
#     'rank': [5],
#     'random_state':[RANDOM_SEED],
#     'l2_reg_w': [7.5],
#     'l2_reg_V': [140],
# }
# time_model = GridSearchCV(als.FMRegression(), param_grid=param_grid, refit=True, verbose=3, n_jobs=-1, cv=5)
# time_model.fit(all_sparse_ft, time_label)
# time_model.best_estimator_

# %%
# splitter = KFold(n_splits=20, shuffle=True)
# mse = []
# for train, test in splitter.split(all_sparse_ft):
#     time_model2 = als.FMRegression(n_iter=10,
#                                 rank=5,
#                                 random_state=RANDOM_SEED,
#                                 l2_reg_w=7.5,
#                                 l2_reg_V=140)
#     time_model2.fit(all_sparse_ft[train], time_label[train])
#     preds = time_model2.predict(all_sparse_ft[test])
#     loss = np.mean((preds - time_label[test])**2)
#     print(loss)
#     mse.append(loss)
# print(f'Overall MSE: {np.mean(mse)}')

# %%
time_model = als.FMRegression(n_iter=10,
                                rank=5,
                                random_state=RANDOM_SEED,
                                l2_reg_w=7.5,
                                l2_reg_V=140)
time_model.fit(all_sparse_ft, time_label)

# %%
model_file = open('fastfm.obj', 'wb')
pickle.dump(time_model, model_file)
model_file.close()

# %%
test_df = pd.read_csv('./pairs_Hours.csv')
testpred = test_df.copy()
test_df['itemID'] = test_df['gameID']
test_df['userIDX'] = user_oe.transform(test_df[['userID']])
test_df['itemIDX'] = item_oe.transform(test_df[['gameID']])
test_df.drop(columns=['gameID', 'prediction'], inplace=True)
test_sparse = convert_sparse_df(test_df, True)
preds = time_model.predict(test_sparse)
testpred = pd.read_csv('./pairs_Hours.csv')
testpred['prediction'] = preds
testpred.to_csv('./predictions_Hours.csv', index=False)


