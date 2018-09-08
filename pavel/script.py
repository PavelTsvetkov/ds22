# run nltk_download.py before running this for first time
import pandas as pd
import collections
from nltk.tokenize import sent_tokenize
from nltk.tokenize import word_tokenize
import numpy as np

from pavel.utils import genres_to_array, rectify_json, count_words

df = pd.read_csv(
    "C:\\tmp\\dabble\\movies_metadata.csv")  # Uses this dataset https://www.kaggle.com/rounakbanik/the-movies-dataset

df['genres'] = df['genres'].apply(rectify_json)

df['genres_arr'] = df['genres'].apply(genres_to_array)

data3 = df['genres_arr'].apply(collections.Counter)

genres_df = pd.DataFrame.from_records(data3).fillna(value=0)

print("Genres:",len(genres_df.columns.values))

word_stats = {}

for overview in df["overview"]:
    try:
        if overview is not None:
            sentences = sent_tokenize(overview);
            for sentence in sentences:
                words = word_tokenize(sentence)
                count_words(words, word_stats)
                # print(words)
        # break
    except TypeError:
        print("Cannot tokenize:", overview)

sorted_d = sorted(((value, key) for (key, value) in word_stats.items()), reverse=True)

word_count=len(sorted_d)

print("Total words:",len(sorted_d))
print("Top 20 words",sorted_d[:20])

word_map={item[1]: index for index, item in enumerate(sorted_d)}

