# run nltk_download.py before running this for first time
import pandas as pd
import collections

from nltk import NaiveBayesClassifier
from nltk.tokenize import sent_tokenize
from nltk.tokenize import word_tokenize
import numpy as np
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import confusion_matrix
from sklearn.neural_network import MLPClassifier

from pavel.utils import genres_to_array, rectify_json, count_words

df = pd.read_csv(
    "C:\\tmp\\dabble\\movies_metadata.csv")  # Uses this dataset https://www.kaggle.com/rounakbanik/the-movies-dataset

df['genres'] = df['genres'].apply(rectify_json)

df['genres_arr'] = df['genres'].apply(genres_to_array)

data3 = df['genres_arr'].apply(collections.Counter)

genres_df = pd.DataFrame.from_records(data3).fillna(value=0)

print("Genres:", len(genres_df.columns.values))

word_stats = {}

df["overview"] = df["overview"].fillna(value="")

# for overview in df["overview"]:
#     try:
#         if overview is not None:
#             sentences = sent_tokenize(overview);
#             for sentence in sentences:
#                 words = word_tokenize(sentence)
#                 count_words(words, word_stats)
#                 # print(words)
#         # break
#     except TypeError:
#         print("Cannot tokenize:", overview)
#
# sorted_d = sorted(((value, key) for (key, value) in word_stats.items()), reverse=True)
#
# word_count = len(sorted_d)
#
# print("Total words:", len(sorted_d))
# print("Top 20 words", sorted_d[:20])

# word_map = {item[1]: index for index, item in enumerate(sorted_d)}

tfid = TfidfVectorizer(lowercase=True, stop_words="english", max_features=5000)

print("transforming input")
input_X = tfid.fit_transform(df["overview"])

target_Y = genres_df[["gen_Action"]].values

train_size = int(0.8 * input_X.shape[0])

print("Train size:", train_size)

train_x = input_X[:train_size]
train_y = target_Y[:train_size]

validation_x = input_X[train_size:]
validation_y = target_Y[train_size:]

rf = MLPClassifier(verbose=True, early_stopping=True)

print("Fitting forest")
rf.fit(train_x,train_y)

pred_y=rf.predict(validation_x)

conf_matr=confusion_matrix(validation_y,pred_y)

print(conf_matr)

