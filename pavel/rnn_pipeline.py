import pandas as pd
import numpy as np
from sklearn.ensemble import AdaBoostClassifier, RandomForestClassifier, VotingClassifier
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score
from sklearn.naive_bayes import GaussianNB
from sklearn.neural_network import MLPClassifier
from sklearn.tree import ExtraTreeClassifier
from keras.callbacks import ModelCheckpoint, TensorBoard
from keras.layers import Dense, Activation, Dropout, Embedding, LSTM, GRU
from keras.models import Sequential

from pavel.keras_utils import f1
from pavel.rnn_constants import *

import pavel.utils as u
from pavel.utils import TFIDFVectorizer, BagOfWordsVectorizer, SequenceVectorizer

# ------------------- Configuration Section ---------------

show_confusion_matr = True

FEATURE_COLUMN = "overview"
CLASS_PREFIX = "gen_"
CLASS_COLUMN = "genres"
pre_process = u.pre_process
detectClasses = u.detectClasses
extract_classes = u.extract_classes

vectorizer = SequenceVectorizer(minDf=0.002, maxDf=0.998, file=DABBLE_VOCAB, maxLen=140)

# ------------------- Configuration Section ---------------

print("Loading dataset")
dataset = pd.read_csv(MOVIES_METADATA_CSV)

print("Preprocessing")
dataset = pre_process(dataset)  # lower case, cleanse, etc.

print("Detecting classes")
dataset,class_count = detectClasses(dataset, column=CLASS_COLUMN, prefix=CLASS_PREFIX)  # generates new columns, one per class

print("Shuffling")
dataset = dataset.sample(frac=1)  # shuflle

print("Splitting into train and test")
train_validation, test = train_test_split(dataset, test_size=0.2)

train, validation = train_test_split(train_validation, test_size=0.2)

print("Training vectorizer")
vectorizer.train(train[FEATURE_COLUMN])
print("Dictionary size:", len(vectorizer.vocab))

print("Vectorizing train input")
train_x = vectorizer.vectorize(train[FEATURE_COLUMN])

print("Extracting train classes")
train_classes, train_y = extract_classes(train, prefx=CLASS_PREFIX, classes=None)

print("Vectorizing test input")
test_x = vectorizer.vectorize(test[FEATURE_COLUMN])

print("Extracting test classes")
tmp_classes, test_y = extract_classes(test, prefx=CLASS_PREFIX, classes=None)

print("Vectorizing validation input")
valid_x = vectorizer.vectorize(validation[FEATURE_COLUMN])

print("Extracting validation classes")
tmp_classes, valid_y = extract_classes(validation, prefx=CLASS_PREFIX, classes=None)

dataset = None
test = None
train = None
train_validation = None
validation = None

print("Saving data")
np.savez_compressed(NUMPY_DATASET, train_x=train_x, train_y=train_y, test_x=test_x, test_y=test_y, valid_x=valid_x,
                    valid_y=valid_y)

mdl = Sequential()
mdl.add(Embedding((len(vectorizer.vocab) + 1), 300, input_length=vectorizer.maxLen))
mdl.add(GRU(32,activation="relu"))

mdl.add(Dense(train_y.shape[1], activation="sigmoid"))

mdl.compile(loss='mean_squared_error', optimizer='rmsprop', metrics=[f1])

print(mdl.input_shape)
mdl.summary()

checkpointer = ModelCheckpoint(filepath=SAVED_MODEL, verbose=1, save_best_only=True)

history = mdl.fit(train_x, train_y, epochs=1000, verbose=1, validation_data=(valid_x, valid_y),
                  batch_size=100, callbacks=[checkpointer])
