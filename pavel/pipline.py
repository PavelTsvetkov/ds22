import pandas as pd
from sklearn.ensemble import AdaBoostClassifier, RandomForestClassifier, VotingClassifier
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score
from sklearn.naive_bayes import GaussianNB
from sklearn.neural_network import MLPClassifier
from sklearn.tree import ExtraTreeClassifier

import pavel.utils as u
from pavel.utils import TFIDFVectorizer, BagOfWordsVectorizer

# ------------------- Configuration Section ---------------
show_confusion_matr = True

FEATURE_COLUMN = "overview"
CLASS_PREFIX = "gen_"
CLASS_COLUMN = "genres"
pre_process = u.pre_process
detectClasses = u.detectClasses
extract_classes = u.extract_classes

classifier = MLPClassifier(verbose=True, early_stopping=True, max_iter=10, hidden_layer_sizes=(300, 300),
                           tol=0.000001)  # F1=0.50
# classifier = RandomForestClassifier(max_depth=1000,n_jobs=4,n_estimators=20)  # F1=0.30
# classifier = ExtraTreeClassifier(max_depth=1000)  # F1=0.32
# classifier = GaussianNB() #not working with simultaneous multiclass
# vectorizer = TFIDFVectorizer(mx_features=10000)
vectorizer = BagOfWordsVectorizer(mx_features=None, n_gram_range=(1, 1), minDf=10, maxDF=0.98)

# ------------------- Configuration Section ---------------

print("Loading dataset")
dataset = pd.read_csv("C:\\tmp\\dabble\\movies_metadata.csv")

print("Preprocessing")
dataset = pre_process(dataset)  # lower case, cleanse, etc.

print("Detecting classes")
dataset,class_count = detectClasses(dataset, column=CLASS_COLUMN, prefix=CLASS_PREFIX)  # generates new columns, one per class

print("Shuffling")
dataset = dataset.sample(frac=1)  # shuflle

print("Splitting into train and test")
train, test = train_test_split(dataset, test_size=0.2)

print("Training vectorizer")
vectorizer.train(train[FEATURE_COLUMN])

print("Vectorizing train input")
train_x = vectorizer.vectorize(train[FEATURE_COLUMN])
print("Train shape:", train_x.shape)

print("Extracting train classes")
train_classes, train_y = extract_classes(train, prefx=CLASS_PREFIX, classes=None)

print("Vectorizing test input")
test_x = vectorizer.vectorize(test[FEATURE_COLUMN])
print("Test shape:", test_x.shape)

print("Extracting test classes")
tmp_classes, test_y = extract_classes(test, prefx=CLASS_PREFIX, classes=None)

dataset = None
test = None
train = None

print("Training classifier")
classifier.fit(train_x.toarray(), train_y)

print("Testing classifier")
pred_y = classifier.predict(test_x.toarray())

print("Confusion matrices")
for i, c in enumerate(train_classes):
    print("Class:", c, "F1 score:", f1_score(test_y[:, i], pred_y[:, i]))
    if (show_confusion_matr):
        conf_matr = confusion_matrix(test_y[:, i], pred_y[:, i])
        print(conf_matr)

print("Total F1 score:", f1_score(test_y, pred_y, average='micro'))

features = classifier.feature_importances_
words = vectorizer.vec.vocabulary_

sorted_w = sorted(zip(features, words), reverse=True, key=lambda item: item[0])
print(sorted_w[:10])
