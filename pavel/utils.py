import collections
import json
from collections import __init__

import pandas as pd
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer

MIN_WORD_LEN = 3

custom_stopwords = set([',', '.', "'s", ')', '(', '``', "''", '’'])

stopwords = set(stopwords.words('english')).union(custom_stopwords)


def genres_to_array(keywords, pref="gen_"):
    return [pref + x['name'] for x in keywords]


def lowercase(s):
    return s.lower()


def rectify_json(s):
    return json.loads(s.replace("'", '"'))


def column_to_set(df, column):
    all_genres = set()
    for c in df[column]:
        split = c.split(",")
        split = [s.strip() for s in split]
        all_genres |= set(split)
    return all_genres


def count_words(words, word_counts):
    for w in words:
        w = w.lower()
        if (w in stopwords or len(w) < MIN_WORD_LEN):
            continue
        if (not w in word_counts):
            word_counts[w] = 1
        else:
            word_counts[w] = word_counts[w] + 1


class FeatureVectorizer(object):
    def __init__(self) -> None:
        super().__init__()

    def vectorize(self, column):
        pass

    def train(self, column):
        pass


class TFIDFVectorizer(FeatureVectorizer):
    def __init__(self) -> None:
        super().__init__()
        self.tfid = TfidfVectorizer(stop_words="english", max_features=5000)

    def vectorize(self, column):
        return self.tfid.transform(column)

    def train(self, column):
        self.tfid.fit(column)


def detectClasses(df, column=None, prefix=None):
    tmp = df[column].apply(genres_to_array)
    data3 = tmp.apply(collections.Counter)
    tmpDF = pd.DataFrame.from_records(data3).fillna(value=0)
    return pd.concat([df, tmpDF], axis=1)


def pre_process(df):
    df['genres'] = df['genres'].apply(rectify_json)
    df["overview"] = df["overview"].fillna(value="")
    df['overview'] = df['overview'].apply(lowercase)


def extract_classes(data, prefx, classes):
    if (classes is not None):
        cols = [prefx + x for x in classes]
    else:
        cols = [col for col in data.columns if col.startswith(prefx)]
    return cols, data.as_matrix(columns=cols)
