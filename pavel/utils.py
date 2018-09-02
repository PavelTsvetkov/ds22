#run nltk_download.py before running this for first time


import pandas as pd
import nltk as nl
from nltk.tokenize import sent_tokenize
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords

english_stops = set(stopwords.words('english'))

df = pd.read_csv("C:\\tmp\\dabble\\movies_metadata.csv")


# Uses this dataset https://www.kaggle.com/rounakbanik/the-movies-dataset


def column_to_set(column):
    all_genres = set()
    for c in df[column]:
        split = c.split(",")
        split = [s.strip() for s in split]
        all_genres |= set(split)
    return all_genres


word_counts = {}


def count_words(words):
    for w in words:
        w = w.lower()
        if (w in english_stops):
            continue
        if (not w in word_counts):
            word_counts[w] = 1
        else:
            word_counts[w] = word_counts[w] + 1


for overview in df["overview"]:
    try:
        if overview is not None:
            sentences = sent_tokenize(overview);
            for sentence in sentences:
                words = word_tokenize(sentence)
                count_words(words)
                # print(words)
        # break
    except TypeError:
        print("Cannot tokenize:", overview)

sorted_d = sorted(((value, key) for (key, value) in word_counts.items()), reverse=True)

print(sorted_d[:20])
