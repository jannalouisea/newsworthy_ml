import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.feature_selection import chi2
from helper import process_text
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.model_selection import ShuffleSplit
from sklearn import svm
import pickle

df = pd.read_csv('all_articles.csv')

df['processed_text'] = df['text'].apply(process_text)
df['processed_text'] = df['processed_text'].apply(lambda x: [' '.join(x)])
df['processed_text'] = df['processed_text'].str[0]

ngram_range = (1,2)
min_df = 10
max_df = .70
max_features = 1000

tfidf = TfidfVectorizer(
    ngram_range=ngram_range,
    stop_words=None,
    lowercase=False,
    max_df=max_df,
    min_df=min_df,
    max_features=max_features,
    sublinear_tf=True
)

features_train = tfidf.fit_transform(df['processed_text'])
features_train = features_train.toarray()
labels_train = df['category']
labels_train = labels_train.astype('int')

# model training
# create the parameter grid based on the results of random search 
C = [.0001, .001, .01, .1]
degree = [3, 4, 5]
gamma = [1, 10, 100]
probability = [True]

param_grid = [
  {'C': C, 'kernel':['linear'], 'probability':probability},
  {'C': C, 'kernel':['poly'], 'degree':degree, 'probability':probability},
  {'C': C, 'kernel':['rbf'], 'gamma':gamma, 'probability':probability}
]

# base model
svc = svm.SVC(random_state=8)

# manually create the splits in CV in order to be able to fix a random_state 
cv_sets = ShuffleSplit(n_splits = 3, test_size = .33, random_state = 8)

# instantiate the grid search model
grid_search = GridSearchCV(
    estimator=svc, 
    param_grid=param_grid,
    scoring='accuracy',
    cv=cv_sets,
    verbose=1
)

# fit the grid search to the data
grid_search.fit(features_train, labels_train)

best_svc = grid_search.best_estimator_
best_svc.fit(features_train, labels_train)

with open('svc_model.pickle', 'wb') as output:
    pickle.dump(best_svc, output)

with open('svc_tfidf.pickle', 'wb') as output:
    pickle.dump(tfidf, output)
