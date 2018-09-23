import collections
import json
from collections import __init__
import numpy as np
import pandas as pd
from nltk.corpus import stopwords
from scipy.sparse import csc_matrix, lil_matrix
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from nltk.tokenize import sent_tokenize
from nltk.tokenize import word_tokenize
import pickle as pic
import os
import gensim

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


def count_words_in_column(column):
    result = {}

    for overview in column:
        try:
            if overview is not None:
                sentences = sent_tokenize(overview);
                for sentence in sentences:
                    words = word_tokenize(sentence)
                    count_words(words, result)
                    # print(words)
            # break
        except TypeError:
            print("Cannot tokenize:", overview)
    return result


def token_gen(strng):
    sentences = sent_tokenize(strng)
    for sentence in sentences:
        for word in word_tokenize(sentence):
            if not (word in stopwords or len(word) < MIN_WORD_LEN):
                yield word
        yield "EOS"


class FeatureVectorizer(object):
    def __init__(self) -> None:
        super().__init__()

    def vectorize(self, column):
        pass

    def train(self, column):
        pass


def count_occurences(word, list):
    i = 0
    for item in list:
        if word in item:
            i = i + 1
    return i


keywords = set(["EOS", "UNK"])


class GensimVectorizer(FeatureVectorizer):

    def __init__(self, model_file, maxLen=300) -> None:
        super().__init__()
        self.maxLen = maxLen
        self.mdl = gensim.models.KeyedVectors.load_word2vec_format(model_file, binary=True)

    def vectorize(self, column):
        result = np.full((len(column), self.maxLen, self.mdl.vector_size), 0)
        for i, item in enumerate(column):
            for k, token in enumerate(token_gen(item)):
                if token in self.mdl:
                    result[i, k] = self.mdl[token]

        return result

    def train(self, column):
        pass


class SequenceVectorizer(FeatureVectorizer):

    def __init__(self, file=None, minDf=0.0, maxDf=1.0, maxLen=300) -> None:
        super().__init__()
        self.maxLen = maxLen
        self.maxDf = maxDf
        self.minDf = minDf
        self.file = file

    def vectorize(self, column):
        result = np.full((len(column), self.maxLen), len(self.vocab))

        for i, item in enumerate(column):
            max_k = 0
            for k, token in enumerate(token_gen(item)):
                max_k = k
                if token in self.vocab:
                    idx = self.vocab[token]
                else:
                    idx = self.vocab["UNK"]
                result[i, k] = idx
            result[i, self.maxLen - max_k:self.maxLen] = result[i, 0:max_k]
            result[i, 0:self.maxLen - max_k] = len(self.vocab)

        return result

    def train(self, column):
        self.vocab = None

        if self.file is not None:
            self.vocab = self.load_from_file()

        if self.vocab is None:
            allwords = self.register_words(column)
            matr = self.register_word_distribution(allwords, column)
            bc = matr.sum(axis=0)
            minCount = self.minDf * len(column)
            maxCount = self.maxDf * len(column)
            indices_to_keep = self.select_indices(bc, maxCount, minCount)

            words_filtered = set(
                [word for word, idx in allwords.items() if (idx in indices_to_keep or word in keywords)])
            words_filtered.add("UNK")
            self.vocab = {word: idx for idx, word in enumerate(words_filtered)}
            self.save_to_file()

    def select_indices(self, bc, maxCount, minCount):
        indices_to_keep = set()
        it = np.nditer(bc[0], flags=['f_index'])
        while not it.finished:
            if (it[0] >= minCount and it[0] <= maxCount):
                indices_to_keep.add(it.index)
            it.iternext()
        return indices_to_keep

    def register_word_distribution(self, allwords, column):
        matr = lil_matrix((len(column), len(allwords)))
        for idx, item in enumerate(column):
            for token in token_gen(item):
                matr[idx, allwords[token]] = 1
        return matr

    def register_words(self, column):
        allwords = {}
        maxLength = 0
        for item in column:
            for i, token in enumerate(token_gen(item)):
                maxLength = max(maxLength, i)
                if not (token in allwords):
                    allwords[token] = len(allwords)
        print("MaxLen", maxLength)
        return allwords

    def load_from_file(self):
        if os.path.isfile(self.file):
            with open(self.file, mode="rb") as f:
                return pic.load(f)

    def save_to_file(self):
        with open(self.file, mode="wb") as f:
            pic.dump(self.vocab, f, pic.HIGHEST_PROTOCOL)


class TFIDFVectorizer(FeatureVectorizer):
    def __init__(self, mx_features=5000) -> None:
        super().__init__()
        self.tfid = TfidfVectorizer(stop_words="english", max_features=mx_features)

    def vectorize(self, column):
        return self.tfid.transform(column)

    def train(self, column):
        self.tfid.fit(column)


class BagOfWordsVectorizer(FeatureVectorizer):

    def __init__(self, mx_features=5000, minDf=40, maxDf=0.01) -> None:
        super().__init__()
        self.vec = CountVectorizer(stop_words="english", max_features=mx_features, min_df=minDf, max_df=maxDf)

    def vectorize(self, column):
        return self.vec.transform(column)

    def train(self, column):
        self.vec.fit(column)


def detectClasses(df, column=None, prefix=None):
    tmp = df[column].apply(genres_to_array)
    data3 = tmp.apply(collections.Counter)
    tmpDF = pd.DataFrame.from_records(data3).fillna(value=0)
    return pd.concat([df, tmpDF], axis=1), len(tmpDF.columns)


def pre_process(df):
    df = df.dropna(subset=["overview"])
    df = df.reset_index(drop=True)
    df['genres'] = df['genres'].apply(rectify_json)
    df['overview'] = df['overview'].apply(lowercase)
    return df


def extract_classes(data, prefx, classes):
    if (classes is not None):
        cols = [prefx + x for x in classes]
    else:
        cols = [col for col in data.columns if col.startswith(prefx)]
    return cols, data.as_matrix(columns=cols)


def year_to_string(word):
    yr = int(word)
    if yr < 1700:
        return "medieval"
    if yr < 1920:
        return "past"
    if yr > 2020:
        return "future"
    if yr >= 2000:
        return "millenium"
    str_y = word[2]
    return {
        '2': 'twenties',
        '3': 'thirties',
        '4': 'forties',
        '5': 'fifties',
        '6': 'sixties',
        '7': 'seventies',
        '8': 'eighties',
        '9': 'nineties',
    }[str_y]


def is_year(word):
    try:
        yr = int(word)
        return True
    except ValueError:
        return False


class TrainingGenerator(object):
    def __init__(self, maxLen=300, model_file=None, synonim_file=None) -> None:
        super().__init__()
        self.synonim_file = synonim_file
        self.maxLen = maxLen
        self.mdl = gensim.models.KeyedVectors.load(model_file)
        self.vector_size = self.mdl.vector_size
        if (synonim_file is not None):
            with(open(synonim_file)) as sf:
                self.synonyms = {line.strip().split(",")[0]: line.strip().split(",")[1] for line in sf}
        else:
            self.synonyms = None

    def generate(self, dataFrame, text_column, Y_prefix, batch_size, Y_classes=None):
        while True:
            batch = dataFrame.sample(n=batch_size)
            tmp, batch_y = extract_classes(batch, Y_prefix, Y_classes)

            column = batch[text_column]

            batch_x = np.full((len(column), self.maxLen, self.mdl.vector_size + 1), 0.0)
            for i, item in enumerate(column):
                max_k = 0;
                for k, token in enumerate(token_gen(item)):
                    max_k = k
                    token = self.choose_token(token)
                    if token is not None:
                        batch_x[i, k, 0:self.mdl.vector_size] = self.mdl[token]
                    else:
                        batch_x[i, k, self.mdl.vector_size] = 1.0
                batch_x[i, self.maxLen - max_k:self.maxLen] = batch_x[i, 0:max_k]
                batch_x[i, 0:self.maxLen - max_k] = 0.0

            yield batch_x, batch_y

    def hasWord(self, word):
        return word in self.mdl

    def can_convert(self, word):
        return self.choose_token(word) in self.mdl

    def generate_balanced(self, dataFrame, text_column, Y_prefix, batch_size, Y_classes):
        pass

    def choose_token(self, word):
        if self.hasWord(word):
            return word
        norm = word.replace("-", "_").replace(".", "")
        if self.hasWord(norm):
            return norm
        if is_year(word):
            ystr = year_to_string(word)
            if self.hasWord(ystr):
                return ystr
        if self.synonyms is not None and word in self.synonyms:
            syn = self.synonyms[word]
            if self.hasWord(syn):
                return syn
        return None
