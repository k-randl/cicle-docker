import sys
import numpy as np
import pandas as pd
import pickle as pkl
from crepes import WrapClassifier
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

TASK = sys.argv[1] + '-category'

data = pd.read_csv('https://zenodo.org/records/10891602/files/food_recall_incidents.csv', index_col=0)

# select input and label from data:
X = data['title'].to_numpy()
y = data[TASK].to_numpy()

# create train and development sets:
X_train, X_dev, y_train, y_dev = train_test_split(X, y, test_size=.2, shuffle=True, stratify=y)
print('Size of development set:', X_dev.shape)
print('Size of train set:      ', X_train.shape)

# create and train input embedding:
tfidf = TfidfVectorizer().fit(X_train)

# since TfidfVectorizer.transform(...) returns a sparse matrix which 'crepes'
# does not handle well, we use the following utility function to encode our texts:
phi = lambda x: tfidf.transform(x).toarray()

# print a sample of the vocabulary to show that we learned something:
list(tfidf.vocabulary_.keys())[:5]

# create label to one-hot and reverse dictionaries:
id2label = np.unique(y_train)
label2id = {l:i for i, l in enumerate(id2label)}

# show label-id mapping:
print('Label-id mapping:', label2id)

# create a conformal base classifier based on Logistic Regression:
base_classifier = WrapClassifier(LogisticRegression())

# train the base classifier:
base_classifier.fit(
    phi(X_train),
    [label2id[y] for y in y_train]
)

# calibrate the base classifier:
base_classifier.calibrate(
    phi(X_dev),
    [label2id[y] for y in y_dev],
    class_cond=True
)

# save trained and calibrated base-classifier:
with open(TASK + '-lr.pkl', 'wb') as file:
    pkl.dump({
        'phi':        tfidf,
        'classifier': base_classifier,
        'id2label':   id2label,
        'label2id':   label2id,
        'texts':      X_train,
        'labels':     y_train
    }, file)