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
'''
topics = [
    ['water', 'new york', 'film', 'hospit', 'nova scotia'], 
    ['trudeau', 'minist', 'program', 'govern', 'parti', 'prime', 'public'],
    ['mask', 'open', 'peopl', 'health', 'servic', 'covid', 'public', 'distanc'],
    ['case', 'covid', 'health', 'new', 'test', 'number', 'death', 'provinc'],
    ['peopl', 'just', 'work', 'famili', 'make', 'want', 'day', 'home'],
    ['lake', 'fish', 'speci', 'river', 'agricultur', 'wildlif', 'salmon', 'fisheri'],
    ['north', 'road', 'south', 'dog', 'highway', 'area', 'river', 'citi'],
    ['compani', 'cent', 'home', 'sale', 'million', 'price', 'tax', 'month'],
    ['communiti', 'ridg', 'bc', 'mapl', 'citi', 'park', 'facebook'],
    ['langley', 'com', 'artist', 'festiv', 'www', 'perform', 'award', 'facebook'],
    ['car', 'vehicl', 'drive', 'driver', 'engin', 'wheel', 'new', 'seat'],
    ['govern', 'canada', 'pandem', 'covid', 'work', 'canadian', 'busi', 'peopl'],
    ['team', 'player', 'play', 'game', 'season', 'toronto', 'leagu', 'sport'],
    ['polic', 'offic', 'rcmp', 'investig', 'surrey', 'man', 'juli', 'report']
]
topics = [
    ['famili just time peopl year home life day'], # lifestyle
    ['case health test covid new death outbreak hospit'], # covid-19
    ['polic investig man offic charg rcmp vehicl incid'] # local
    ['player team game leagu season play nhl coach'], # sports
    ['busi govern program tax billion pandem fund feder'], # politics
    ['car wheel vehicl engin drive rear litr model'], # buisiness
    ['mask wear face cover mandatori public'], # covid-19
    ['reopen stage restaur ontario open park citi theatr'], # local ?
    ['cent price sale market home real estat hous'], # business
    ['music perform artist art concert film musician song'], # arts & entertainment
    ['ridg mapl editor meadow letter mapleridgenews com'], # local
    ['langley chilliwack rcmp surrey com read facebook local'], # local
    ['flight canada court china trump canadian travel countri'], # politics
    ['trudeau minist chariti prime program ethic student'], # politics
    ['jay blue roger centr season game toronto'] # sports
]
'''

def get_category_name(category_id):
    for category, id_ in category_codes.items():    
        if id_ == category_id:
            return category

# df = pd.read_csv('News_dataset.csv',sep=None,header=None,engine='python')
# df = df.drop([0])
# df = df.rename(columns={0:'filename',1:'Content',2:'category'})
# df['processed_text'] = df['Content'].apply(process_text)
# df['processed_text'] = df['processed_text'].apply(lambda x: [' '.join(x)])
# df['processed_text'] = df['processed_text'].str[0]

df = pd.read_csv('all_articles.csv')
# print(df.head())

df['processed_text'] = df['text'].apply(process_text)
df['processed_text'] = df['processed_text'].apply(lambda x: [' '.join(x)])
df['processed_text'] = df['processed_text'].str[0]


df_topics = pd.read_csv('topics.csv')
# print(df_topics['topics'])
topics = df['topics']
# print(df_topics['topics'])
# print(topics)

# THIS IS A BUG - FIX IT
df_2 = pd.DataFrame(data=df['topics'],columns=['topics'])
df_2 = df_2.drop_duplicates().reset_index().drop(columns=['index'])
print(df_2)

# label coding
category_codes = {
    'business':0,
    'arts & sentertainment':1,
    'politics':2,
    'sports':3,
    'science & tech':4,
    'covid-19':5,
    'lifestyle':6,
    'local':7,
    'health':8,
    'crisis':9
}

X_train, X_test, y_train, y_test = train_test_split(
    df['processed_text'],
    df['category'],
    test_size=0.15,
    random_state=8
)

# print(X_test)

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

features_train = tfidf.fit_transform(X_train)
features_train = features_train.toarray()
labels_train = y_train
labels_train = labels_train.astype('int')
print(features_train.shape)

features_test = tfidf.transform(X_test).toarray()
labels_test = y_test
labels_test = labels_test.astype('int')
print(features_test.shape)

for category, category_id in sorted(category_codes.items()):
    features_chi2 = chi2(features_train, labels_train == category_id) # just labels_train?
    # chi2 statistic of each feature
    indices = np.argsort(features_chi2[0])
    feature_names = np.array(tfidf.get_feature_names())[indices]
    print('{} category:'.format(category))
    print('feature names: {}'.format(feature_names[:5]))
    print()


# model training

# Create the parameter grid based on the results of random search 
C = [.0001, .001, .01, .1]
degree = [3, 4, 5]
gamma = [1, 10, 100]
probability = [True]

param_grid = [
  {'C': C, 'kernel':['linear'], 'probability':probability},
  {'C': C, 'kernel':['poly'], 'degree':degree, 'probability':probability},
  {'C': C, 'kernel':['rbf'], 'gamma':gamma, 'probability':probability}
]

# Create a base model
svc = svm.SVC(random_state=8)

# Manually create the splits in CV in order to be able to fix a random_state (GridSearchCV doesn't have that argument)
cv_sets = ShuffleSplit(n_splits = 3, test_size = .33, random_state = 8)

# Instantiate the grid search model
grid_search = GridSearchCV(
    estimator=svc, 
    param_grid=param_grid,
    scoring='accuracy',
    cv=cv_sets,
    verbose=1
)

# Fit the grid search to the data
grid_search.fit(features_train, labels_train)

best_svc = grid_search.best_estimator_
best_svc.fit(features_train, labels_train)

svc_pred = best_svc.predict(features_test)

print('training accuracy: {}'.format(accuracy_score(labels_train, best_svc.predict(features_train))))
print('testing accuracy: {}'.format(accuracy_score(labels_test, svc_pred)))

categories = []
probabilities = []

for idx, row in df_2.iterrows():
    topic_matrix = tfidf.transform(row)
    topic_matrix = topic_matrix.toarray()
    svc_pred_2 = best_svc.predict(topic_matrix)[0]
    svc_pred_2_proba = best_svc.predict_proba(topic_matrix)[0]
    category = get_category_name(svc_pred_2)
    probability = svc_pred_2_proba.max()*100
    print('topic: {}'.format(row['topics']))
    print('category: {}'.format(category))
    print('probability: {}\n'.format(probability))
    categories.append(category)
    probabilities.append(probability)



df_topics['category'] = categories
df_topics['probability'] = probabilities
df_topics.to_csv('/Users/miya/Documents/GitHub/ai4good_news/news_project/topics_with_category.csv',header=True)
