import pandas as pd
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split

# ------------------- Configuration Section ---------------
from sklearn.neural_network import MLPClassifier

import pavel.utils as u

from pavel.utils import TFIDFVectorizer

FEATURE_COLUMN = "overview"
CLASS_PREFIX = "gen_"
CLASS_COLUMN = "genres"
pre_process = u.pre_process
detectClasses = u.detectClasses
extract_classes = u.extract_classes

# ------------------- Configuration Section ---------------

print("Loading dataset")
dataset = pd.read_csv("C:\\tmp\\dabble\\movies_metadata.csv")

print("Preprocessing")
pre_process(dataset)  # lower case, cleanse, etc.

print("Detecting classes")
dataset = detectClasses(dataset, column=CLASS_COLUMN, prefix=CLASS_PREFIX)  # generates new columns, one per class

print("Shuffling")
dataset = dataset.sample(frac=1)  # shuflle

print("Splitting into train and test")
train, test = train_test_split(dataset, test_size=0.2)

print("Creating vectorizer")
vectorizer = TFIDFVectorizer(train[FEATURE_COLUMN])

print("Vectorizing train input")
train_x = vectorizer.vectorize(train[FEATURE_COLUMN])
print("Train shape:", train_x.shape)

print("Extracting train classes")
train_classes, train_y = extract_classes(train, prefx=CLASS_PREFIX, classes=None)

print("Vectorizing test input")
test_x = vectorizer.vectorize(train[FEATURE_COLUMN])

print("Extracting test classes")
tmp_classes, test_y = extract_classes(train, prefx=CLASS_PREFIX, classes=None)

print("Creating classifier")
classifier = MLPClassifier(verbose=True, early_stopping=True, max_iter=10, hidden_layer_sizes=(100, 100), tol=0.000001)

print("Training classifier")
classifier.fit(train_x, train_y)

print("Testing classifier")
pred_y = classifier.predict(test_x)

print("Confusion matrces")
for i, c in enumerate(train_classes):
    print("Class:", c)
    conf_matr = confusion_matrix(test_y[:, i], pred_y[:, i])
    print(conf_matr)
