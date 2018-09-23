import math

import numpy as np
from sklearn.metrics import f1_score

from pavel.keras_utils import f1
from pavel.rnn_constants import *
from keras.models import load_model

print("Loading data")
dataset = np.load(NUMPY_DATASET)

test_x = dataset["test_x"]
test_y = dataset["test_y"]

train_x = dataset["train_x"]
train_y = dataset["train_y"]

print("Loading model")
mdl = load_model(SAVED_MODEL, custom_objects={"f1": f1})

print("Compiling")
mdl.compile(loss='binary_crossentropy', optimizer='rmsprop', metrics=["accuracy"])
mdl.summary()

print("Predicting on train")
pred_y = mdl.predict(train_x, batch_size=100)

pred_y = np.round(pred_y)

print("Total F1 score:", f1_score(train_y, pred_y, average='micro'))

print("Predicting on test")
pred_y = mdl.predict(test_x, batch_size=100)

pred_y = np.round(pred_y)

print("Total F1 score:", f1_score(test_y, pred_y, average='micro'))
