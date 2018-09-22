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
from keras.layers import Dense, Activation, Dropout, Embedding, LSTM, GRU, BatchNormalization
from keras.models import Sequential

from pavel.keras_utils import f1
from pavel.rnn_constants import *

import pavel.utils as u
from pavel.utils import TFIDFVectorizer, BagOfWordsVectorizer, SequenceVectorizer, GensimVectorizer, TrainingGenerator

# ------------------- Configuration Section ---------------

show_confusion_matr = True

FEATURE_COLUMN = "overview"
CLASS_PREFIX = "gen_"
CLASS_COLUMN = "genres"
pre_process = u.pre_process
detectClasses = u.detectClasses
extract_classes = u.extract_classes

only_classes = None# ["Drama"]

# generator = TrainingGenerator(maxLen=140, model_file="C:\\tmp\\dabble\\GoogleNews-vectors-negative300.bin")
generator = TrainingGenerator(maxLen=140, model_file="C:\\tmp\\dabble\\GoogleNews-kvectors.bin")

# ------------------- Configuration Section ---------------

print("Loading dataset")
dataset = pd.read_csv(MOVIES_METADATA_CSV)

print("Preprocessing")
dataset = pre_process(dataset)  # lower case, cleanse, etc.

print("Detecting classes")
dataset, class_count = detectClasses(dataset, column=CLASS_COLUMN,
                                     prefix=CLASS_PREFIX)  # generates new columns, one per class

if only_classes is not None:
    class_count = 1

print("Shuffling")
dataset = dataset.sample(frac=1)  # shuflle

print("Splitting into train and test")
train_validation, test = train_test_split(dataset, test_size=0.2)

train, validation = train_test_split(train_validation, test_size=0.2)

dataset = None
test = None
# train = None
train_validation = None
# validation = None

mdl = Sequential()
mdl.add(LSTM(50, activation="tanh", input_shape=(generator.maxLen, generator.vector_size + 1),recurrent_activation="tanh"))
mdl.add(BatchNormalization())
mdl.add(Dense(100, activation="selu"))
mdl.add(BatchNormalization())
mdl.add(Dense(class_count, activation="sigmoid"))

mdl.compile(loss='binary_crossentropy', optimizer='adam', metrics=[f1,"accuracy"])

print(mdl.input_shape)
mdl.summary()

checkpointer = ModelCheckpoint(filepath=SAVED_MODEL, verbose=1, save_best_only=True)

train_gen = generator.generate(train, FEATURE_COLUMN, CLASS_PREFIX, 100, Y_classes=only_classes)
valid_gen = generator.generate(validation, FEATURE_COLUMN, CLASS_PREFIX, 100, Y_classes=only_classes)

history = mdl.fit_generator(train_gen, steps_per_epoch=30, epochs=1000, verbose=1, validation_data=valid_gen,
                            validation_steps=60, callbacks=[checkpointer])
