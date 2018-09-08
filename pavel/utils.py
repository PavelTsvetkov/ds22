import json
from nltk.corpus import stopwords

MIN_WORD_LEN = 3

custom_stopwords = set([',', '.', "'s", ')', '(', '``', "''", '’'])

stopwords = set(stopwords.words('english')).union(custom_stopwords)


def genres_to_array(keywords):
    return ["gen_" + x['name'] for x in keywords]


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
