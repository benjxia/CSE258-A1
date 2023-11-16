# %% [markdown]
# # CSE 258: Assignment 1
# ### Benjamin Xia

# %% [markdown]
# ### Setup

# %%
import numpy as np
import pandas as pd
from scipy import sparse

from sklearn import preprocessing
from sklearn.decomposition import LatentDirichletAllocation
from sklearn import feature_extraction
from sklearn.model_selection import KFold

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

test = False

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
time_played = defaultdict(dict)
item_mean_hr = defaultdict()
user_mean_hr = defaultdict()
ft = ['early_access', 'compensation'] # features unavailable/cannot be approximated in inference
def read_json(path):
    f: gzip.GzipFile = gzip.open(path)
    f.readline()
    for line in f:
        entry = eval(line)
        yield entry

# Encode userID and itemID as integers
def process_data():
    global itemset, userset, U, I, user_mean_hr, item_mean_hr
    data = []
    for entry in read_json('train.json.gz'):
        data.append(entry)
        time_played[entry['userID']][entry['gameID']] = entry['hours_transformed']

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
    item_mean_hr = dict(df.groupby('itemID')['hours_transformed'].mean())
    user_mean_hr = dict(df.groupby('userID')['hours_transformed'].mean())
    return df, time_label

df, time_label = process_data()
user_mean_ft = df.groupby('userIDX')[ft].mean()
item_mean_ft = df.groupby('itemIDX')[ft].mean()
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
# text_embed = text_embed / np.linalg.norm(text_embed, axis=1)[...,None]

df.drop('text', axis=1, inplace=True)


# %%
text_embed = np.concatenate((np.arange(0, len(text_embed))[:,  None], text_embed, item_mean_ft.to_numpy()), axis=1)

# %%
df_train = df.iloc[:150000]
df_time_train_label = time_label[:150000]
df_valid = df.iloc[150000:]
df_time_valid_label = time_label[150000:]

# %% [markdown]
# ### Played Predictions

# %%
played_model = RankFM(factors=5,
               loss='warp',
               max_samples=300,
               learning_exponent=0.25,
               learning_schedule='invscaling')

# %%
# # Validation stuff - determine factor dimensions
# from sklearn.model_selection import KFold

# kf = KFold(n_splits=10, shuffle=True, random_state=RANDOM_SEED)

# for k in [1, 2, 3, 4, 5, 6, 10, 20]:
#     played_model = RankFM(factors=k,
#                 loss='warp',
#                 max_samples=300,
#                 learning_exponent=0.25,
#                 learning_schedule='invscaling')
#     fold_accs = []
#     for i, (train, test) in enumerate(kf.split(df[['userIDX', 'itemIDX']])):
#         played_model.fit(df.iloc[train][['userIDX', 'itemIDX']], item_features=text_embed, epochs=20, verbose=False)
#         neg_pairs = []
#         for review in df.iloc[test].iterrows():
#             review = review[1]
#             sample = random.sample(itemset.difference(I[review['userID']]), k=1)[0]
#             neg_pairs.append([review['userIDX'], istoi[sample]])
#         pos_pairs = df.iloc[test][['userIDX', 'itemIDX']].to_numpy()
#         neg_pairs = np.array(neg_pairs)
#         pos_scores = played_model.predict(pos_pairs)
#         neg_scores = played_model.predict(neg_pairs)
#         acc = (np.mean(pos_scores >= 0) + np.mean(neg_scores < 0)) / 2
#         fold_accs.append(acc)
#         print(f'Validation %: {acc * 100}')
#     print(f'k: {k} = {np.mean(fold_accs)}')

# %%
# # Determine training epochs
# kf = KFold(n_splits=10, shuffle=True, random_state=RANDOM_SEED)
# if not test:
#     accs = np.zeros((10, 50))
#     for j, (train, test) in enumerate(kf.split(df[['userIDX', 'itemIDX']])):
#         played_model = RankFM(factors=5,
#                 loss='warp',
#                 max_samples=300,
#                 learning_exponent=0.25,
#                 learning_schedule='invscaling')
#         for i in range(50):
#             played_model.fit_partial(df.iloc[train][['userIDX', 'itemIDX']], item_features=text_embed, epochs=4, verbose=False)
#             neg_pairs = []
#             for review in df.iloc[test].iterrows():
#                 review = review[1]
#                 sample = random.sample(itemset.difference(I[review['userID']]), k=1)[0]
#                 neg_pairs.append([review['userIDX'], istoi[sample]])
#             pos_pairs = df.iloc[test][['userIDX', 'itemIDX']].to_numpy()
#             neg_pairs = np.array(neg_pairs)
#             pos_scores = played_model.predict(pos_pairs)
#             neg_scores = played_model.predict(neg_pairs)
#             acc = (np.mean(pos_scores >= 0) + np.mean(neg_scores < 0)) / 2
#             print(f'Validation %: {acc * 100}')
#             accs[j, i] = acc

#     print(accs)

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
played_model = RankFM(factors=5,
               loss='warp',
               max_samples=300,
               learning_exponent=0.25,
               learning_schedule='invscaling')
train = False
save = False
test = True
if train == True:
    best_model = None
    best_acc = 0
    for i in range(50):
        # switch fit_partial's dataframe to df_train for testing, "df" for actual predictions
        if test == True:
            played_model.fit_partial(df[['userIDX', 'itemIDX']], item_features=text_embed, epochs=4, verbose=False)
        else:
            played_model.fit_partial(df_train[['userIDX', 'itemIDX']], item_features=text_embed, epochs=4, verbose=False)
        acc = played_validate(played_model)
        if acc > best_acc:
            best_model = copy.deepcopy(played_model)
            best_acc = acc
    if save == True:
        model_file = open('rankfm.obj', 'wb')
        pickle.dump(best_model, model_file)
        model_file.close()
else:
    model_file = open('rankfm.obj', 'rb')
    best_model = pickle.load(model_file)
    played_model = best_model
    model_file.close()

# %%
popular_games = dict(df['itemID'].value_counts()[:int(.75 * len(df['itemID'].unique()))])

# %% [markdown]
# #### Make and write predictions

# %%
# test_df = pd.read_csv('./pairs_Played.csv')
# testpred = test_df.copy()
# test_df['itemID'] = test_df['gameID']
# # Map unseen entries to default user (this user is already grouped with other users due to their few # of reviews in training set)
# test_df['userID'] = test_df['userID'].map(lambda x: x if x in userset else 'u03473346')
# test_df['userIDX'] = user_oe.transform(test_df[['userID']])
# test_df['itemIDX'] = item_oe.transform(test_df[['gameID']])
# test_df.drop(columns=['gameID', 'prediction'], inplace=True)
# scores = best_model.predict(test_df[['userIDX', 'itemIDX']])
# testpred = pd.read_csv('./pairs_Played.csv')
# testpred['prediction'] = (scores >= np.median(scores)).astype(np.int32)
# testpred.to_csv('./predictions_Played.csv', index=False)

# %% [markdown]
# ### Time Prediction

# %%
def convert_df(df: pd.DataFrame):
    datum = np.zeros((len(df), 10 + 10 + 22))
    for i, (idx, row) in enumerate(df.iterrows()):
        user = row['userIDX']
        item = row['itemIDX']
        datum[i, :10] = played_model.v_u[user]
        datum[i, 10:20] = played_model.v_i[item]
        datum[i, 20:] = text_embed[item, 1:]
    return datum
time_train = convert_df(df_train)
time_valid = convert_df(df_valid)

# %% [markdown]
# #### Collaborative Filtering with played prediction latent factors (this sucks)

# %%
# def lr_sim(item_i, item_j):
#     lr_item_i = best_model.v_i[item_i]
#     lr_item_j = best_model.v_i[item_j]
#     return np.dot(lr_item_i, lr_item_j) / (np.linalg.norm(lr_item_i) * np.linalg.norm(lr_item_j))

# def jaccard_sim(item_i, item_j):
#     s1 = U[item_i]
#     s2 = U[item_j]
#     return len(s1.intersection(s2)) / len(s1.union(s2))

# def cf_predict(user_id, user_idx, item_id, item_idx):
#     sim_sum = 0 # Sum of similarity scores (besides current)
#     output = 0
#     for item_j in time_played[user_id]:
#         if item_j == item_id:
#             continue
#         sim = lr_sim(item_idx, istoi[item_j])
#         # sim = jaccard_sim(item_j, item_id)
#         score = sim * (time_played[user_id][item_j] - item_mean_hr[item_j])
#         output += score
#         sim_sum += np.abs(sim)
#     if sim_sum == 0:
#         return item_mean_hr[item_id]
#     output /= sim_sum
#     output += item_mean_hr[item_id]
#     return output

# preds = np.zeros((len(df_train)))
# for i in range(len(df_train)):
#     row = df_train.iloc[i]
#     preds[i] = cf_predict(row['userID'], row['userIDX'], row['itemID'], row['itemIDX'])
# print(np.mean((preds - df_time_train_label)**2))
# preds = np.zeros((len(df_time_valid_label)))
# for i in range(len(df_valid)):
#     row = df_valid.iloc[i]
#     preds[i] = cf_predict(row['userID'], row['userIDX'], row['itemID'], row['itemIDX'])
# print(np.mean((preds - df_time_valid_label)**2))

# %% [markdown]
# #### XGBoost with played predictioin latent factors (this sucks)

# %%
# import xgboost
# time_model = xgboost.XGBRegressor(n_estimators=5, reg_alpha=1, gamma=1, reg_lambda=1, max_depth=10)
# # time_model = ensemble.RandomForestRegressor(n_estimators=10, max_depth=10, max_features='sqrt', n_jobs=-1)
# time_model.fit(time_train, df_time_train_label)

# train_preds = time_model.predict(time_train)
# # train_preds[train_preds < 0] = 0
# # train_preds[train_preds > 14] = 14
# print(np.mean((train_preds - df_time_train_label)**2))
# valid_preds = time_model.predict(time_valid)
# # valid_preds[valid_preds < 0] = 0
# # valid_preds[valid_preds > 14] = 14
# MSE = np.mean((valid_preds - df_time_valid_label)**2)
# print(MSE)

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
# time_train = convert_sparse_df(df_train, False)
# time_valid = convert_sparse_df(df_valid, False)


# %%
all_sparse_ft = convert_sparse_df(df, True)

# %%
splitter = KFold(n_splits=20, shuffle=True)
mse = []
for train, test in splitter.split(all_sparse):
    time_model = als.FMRegression(n_iter=20,
                                rank=4,
                                init_stdev=0,
                                random_state=RANDOM_SEED,
                                l2_reg_w=5,
                                l2_reg_V=200)
    time_model.fit(all_sparse_ft[train], time_label[train])
    preds = time_model.predict(all_sparse_ft[test])
    loss = np.mean((preds - time_label[test])**2)
    print(loss)
    mse.append(loss)
print(f'Overall MSE: {np.mean(mse)}')

# %%
time_model = als.FMRegression(n_iter=20,
                            rank=4,
                            init_stdev=0,
                            random_state=RANDOM_SEED,
                            l2_reg_w=5,
                            l2_reg_V=200)
time_model.fit(all_sparse_ft, time_label)

# %%
model_file = open('fastfm.obj', 'wb')
pickle.dump(time_model, model_file)
model_file.close()

# %%
test_df = pd.read_csv('./pairs_Hours.csv')
testpred = test_df.copy()
test_df['itemID'] = test_df['gameID']
# Map unseen entries to default user (this user is already grouped with other users due to their few # of reviews in training set)
test_df['userID'] = test_df['userID'].map(lambda x: x if x in userset else 'u03473346')
test_df['userIDX'] = user_oe.transform(test_df[['userID']])
test_df['itemIDX'] = item_oe.transform(test_df[['gameID']])
test_df.drop(columns=['gameID', 'prediction'], inplace=True)
test_sparse = convert_sparse_df(test_df, True)
preds = time_model.predict(test_sparse)
testpred = pd.read_csv('./pairs_Hours.csv')
testpred['prediction'] = preds
testpred.to_csv('./predictions_Hours.csv', index=False)

# %% [markdown]
# #### HW3 Modified $\alpha + \beta_u + \beta_i$

# %%
def readJSON(path):
    f = gzip.open(path, 'rt', encoding='utf8')
    f.readline()
    for l in f:
        d = eval(l)
        u = d['userID']
        g = d['gameID']
        yield u,g,d

allHours = []
for l in readJSON("train.json.gz"):
    allHours.append(l)

hoursTrain = allHours[:165000]
hoursValid = allHours[165000:]

# Any other preprocessing...
itemset = set()
userset = set()
user_stoi = dict()
user_itos = []
item_stoi = dict()
item_itos = []
for user, item, review in allHours:
    itemset.add(item)
    userset.add(item)
    if user not in user_stoi:
        user_stoi[user] = len(user_itos)
        user_itos.append(user)
    if item not in item_stoi:
        item_stoi[item] = len(item_itos)
        item_itos.append(item)


U = defaultdict(set)
I = defaultdict(set)
validPairs_part_1 = []
for review in hoursTrain:
    user = review[0]
    item = review[1]
    U[item].add(user)
    I[user].add(item)

I_arr = np.array([len(I[user_itos[u]]) for u in range(len(I))])
U_arr = np.array([len(U[item_itos[i]]) for i in range(len(U))])

validPairs_part_1 = [[user_stoi[user], item_stoi[item]] for user, item, review_body in hoursValid]
validLabels_part_1 = np.array([1] * len(hoursValid) + [0] * len(hoursValid))

validPairs_part_2 = validPairs_part_1.copy()
validPairs_part_2 = np.array(validPairs_part_2)
validLabels_part_2 = np.array([review['hours_transformed'] for user, item, review in hoursValid])

# Construct a new validation set w/ negative pairs
for user, item, review in hoursValid:
    sample = random.sample(itemset.difference(I[user]), 1)[0]
    validPairs_part_1.append([user_stoi[user], item_stoi[sample]])

validPairs_part_1 = np.array(validPairs_part_1)

# %%
trainHours = np.array([r[2]['hours_transformed'] for r in hoursTrain])
globalAverage = sum(trainHours) * 1.0 / len(trainHours)
trainPairs = np.array([[user_stoi[user], item_stoi[item]] for user, item, review in hoursTrain])
allPairs = np.array([[user_stoi[user], item_stoi[item]] for user, item, review in allHours])
allHours = np.array([r[2]['hours_transformed'] for r in allHours])

# %%
def closed_form(lamb, alpha, beta_u, beta_i, train_label, train_pair):
    new_beta_u = np.zeros_like(beta_u)
    new_beta_i = np.zeros_like(beta_i)
    alpha = np.mean(train_label - beta_u[train_pair[:, 0]] - beta_i[train_pair[:, 1]])
    delta = (train_label - alpha - beta_i[train_pair[:, 1]]) / (lamb + I_arr[train_pair[:, 0]])
    for i in range(len(train_pair)):
        new_beta_u[train_pair[i, 0]] += delta[i]
    beta_u = new_beta_u
    delta = (train_label - alpha - beta_u[train_pair[:, 0]]) / (lamb + U_arr[train_pair[:, 1]])
    for i in range(len(train_pair)):
        new_beta_i[train_pair[i, 1]] += delta[i]
    beta_i = new_beta_i
    return alpha, beta_u, beta_i

# %%
beta_u = np.zeros(len(I))
beta_i = np.zeros(len(U))
alpha = globalAverage # Could initialize anywhere, this is a guess

for i in tqdm(range(200)):
    alpha, beta_u, beta_i = closed_form(5, alpha, beta_u, beta_i, trainHours, trainPairs)

validMSE = 0
for i, (user, item) in enumerate(validPairs_part_2):
    validMSE += (validLabels_part_2[i] - alpha - beta_u[user] - beta_i[item]) ** 2
validMSE /= len(validPairs_part_2)
print(validMSE)

# %%
validPairs_part_2

# %%
splitter = KFold(n_splits=20, shuffle=True, random_state=RANDOM_SEED)
mselist = []
for train, test in splitter.split(allPairs):
    beta_u = np.zeros(len(I))
    beta_i = np.zeros(len(U))
    alpha = globalAverage # Could initialize anywhere, this is a guess

    for i in tqdm(range(300)):
        alpha, beta_u, beta_i = closed_form(5, alpha, beta_u, beta_i, allHours[train], allPairs[train])
    validMSE = 0
    for i, (user, item) in enumerate(allPairs[test]):
        validMSE += (allHours[test][i] - alpha - beta_u[user] - beta_i[item]) ** 2
    validMSE /= len(test)
    mselist.append(validMSE)
    print(validMSE)
print(f'Overall MSE: {np.mean(mselist)}')

# %%
splitter = KFold(n_splits=10, shuffle=True)
for train, test in splitter.split(all_sparse):
    time_model = als.FMRegression(n_iter=5,
                                rank=0,
                                init_stdev=0,
                                random_state=RANDOM_SEED,
                                l2_reg_w=0.1,
                                l2_reg_V=0)
    time_model.fit(all_sparse[train], time_label[train])
    preds = time_model.predict(all_sparse[test])
    print(np.mean((preds - time_label[test])**2))

# %% [markdown]
# #### Make and write predictions

# %%
# test_df = pd.read_csv('./pairs_Hours.csv')
# testpred = test_df.copy()
# test_df['itemID'] = test_df['gameID']
# # Map unseen entries to default user (this user is already grouped with other users due to their few # of reviews in training set)
# test_df['userID'] = test_df['userID'].map(lambda x: x if x in userset else 'u03473346')
# test_df['userIDX'] = user_oe.transform(test_df[['userID']])
# test_df['itemIDX'] = item_oe.transform(test_df[['gameID']])
# test_df.drop(columns=['gameID', 'prediction'], inplace=True)

# time_test = convert_df(test_df)
# preds = time_model.predict(time_test)

# testpred = pd.read_csv('./pairs_Hours.csv')
# testpred['prediction'] = preds
# testpred.to_csv('./predictions_Hours.csv', index=False)


