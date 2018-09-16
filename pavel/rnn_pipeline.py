import pandas as pd
from sklearn.ensemble import AdaBoostClassifier, RandomForestClassifier, VotingClassifier
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score
from sklearn.naive_bayes import GaussianNB
from sklearn.neural_network import MLPClassifier
from sklearn.tree import ExtraTreeClassifier
from keras.callbacks import ModelCheckpoint, TensorBoard
from keras.layers import Dense, Activation, Dropout, Embedding, LSTM
from keras.models import Sequential

import pavel.utils as u
from pavel.utils import TFIDFVectorizer, BagOfWordsVectorizer, SequenceVectorizer

# ------------------- Configuration Section ---------------
DABBLE_VOCAB = "C:\\tmp\\dabble\\vocab.dic"
TRAINSET_FILE = "C:\\tmp\\dabble\\trainset.bin"
MOVIES_METADATA_CSV = "C:\\tmp\\dabble\\movies_metadata.csv"
show_confusion_matr = True

FEATURE_COLUMN = "overview"
CLASS_PREFIX = "gen_"
CLASS_COLUMN = "genres"
pre_process = u.pre_process
detectClasses = u.detectClasses
extract_classes = u.extract_classes

vectorizer = SequenceVectorizer(minDf=0.02, maxDf=0.98, file=DABBLE_VOCAB)

# ------------------- Configuration Section ---------------

print("Loading dataset")
dataset = pd.read_csv(MOVIES_METADATA_CSV)

print("Preprocessing")
dataset = pre_process(dataset[:100])  # lower case, cleanse, etc.

print("Detecting classes")
dataset = detectClasses(dataset, column=CLASS_COLUMN, prefix=CLASS_PREFIX)  # generates new columns, one per class

print("Shuffling")
dataset = dataset.sample(frac=1)  # shuflle

print("Splitting into train and test")
train, test = train_test_split(dataset, test_size=0.2)

print("Training vectorizer")
vectorizer.train(train[FEATURE_COLUMN])

print("Vectorizing train input")
train_x = vectorizer.vectorize(train[FEATURE_COLUMN])

print("Extracting train classes")
train_classes, train_y = extract_classes(train, prefx=CLASS_PREFIX, classes=None)

print("Vectorizing test input")
test_x = vectorizer.vectorize(test[FEATURE_COLUMN])

print("Extracting test classes")
tmp_classes, test_y = extract_classes(test, prefx=CLASS_PREFIX, classes=None)

dataset = None
test = None
train = None

mdl = Sequential()
mdl.add(Embedding((len(vectorizer.vocab)+1), 300, input_length=vectorizer.maxLen))
mdl.add(LSTM(100))

mdl.add(Dense(train_y.shape[1], activation="sigmoid"))

mdl.compile(loss='categorical_crossentropy', optimizer='adam', metrics=["accuracy"])

print(mdl.input_shape)
mdl.summary()

history = mdl.fit(train_x, train_y, epochs=1000, verbose=1, validation_split=0.2,
                  batch_size=7)
