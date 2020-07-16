import pandas as pd
import numpy as np
from sklearn.decomposition import NMF
from sklearn.feature_extraction.text import TfidfVectorizer
from gensim.models.coherencemodel import CoherenceModel
from gensim.corpora.dictionary import Dictionary
from gensim.models.nmf import Nmf
from collections import Counter
from operator import itemgetter
import matplotlib.pyplot as plt
import seaborn as sns
sns.set_style('darkgrid')

from helper import word_count, process_text, topic_table, whitespace_tokenizer, unique_words

# parameters
# dictionary
no_below = 5 # words must appear in >=3 articles
no_above = 0.80 # words can't appear in >85% of articles
keep_n = 5000 # keep top 5000 of remaining words
# gensim-coherence nmf
chunksize = 2000
passes = 5
kappa = 0.1
min_prob = 0.01
w_max_iter = 300
w_stop_condition = 0.0001
h_max_iter = 100
h_stop_condition = 0.001
eval_every = True
# tf-idf
min_df = 3
max_df = 0.85
max_features = 5000
# sklearn-model nmf
max_iter = 500
l1_ratio = 0.0
alpha = 0.0
tol = 0.0001

path = '/Users/miya/Documents/GitHub/ai4good_news/news_project/test_nmf_1/'
test_path = 'test_nmf_1/'

# read in data
df = pd.read_csv('clean.csv',engine='python')
df.drop_duplicates(subset=['url'],keep='first',inplace=True)

# create new column 'word_count'
df['word_count'] = df['text'].apply(word_count)

# create new column 'processed_text'
df['processed_text'] = df['text'].apply(process_text)

# flatten list of lists
p_text = df['processed_text']
p_text = [item for sublist in p_text for item in sublist]

# count frequency of word in all documents
top_20 = pd.DataFrame(
    Counter(p_text).most_common(20),
    columns=['word','frequency']
)
num_unique_words = len(set(p_text))

# create dictionary to pass as input to gensim model
texts = df['processed_text']
dictionary = Dictionary(texts)

# filter out words 
dictionary.filter_extremes(
    no_below=no_below,
    no_above=no_above,
    keep_n=keep_n
)

# convert to bag of words (corpus)
# need corpus to pass as input to gensim nmf model
# [[(word_id, # times word appears in document),...],...]
corpus = [dictionary.doc2bow(text) for text in texts]
topic_nums = list(np.arange(5,75+1,5))

# find optimal # of topics
# use gensim nmf model so we can pass as input to gensim CoherenceModel
coherence_scores = []
for num in topic_nums:
    # initialize NMF model
    # gensim model
    nmf = Nmf(
        corpus=corpus,
        num_topics=num,
        id2word=dictionary, # mapping from word IDs to words
        chunksize=chunksize, # number of documents to be used in each training chunk
        passes=passes, # number of full passes over the training corpus
        kappa=kappa, # gradient descent step size (larger => model trains faster but could lead to non-convergence if set too large)
        minimum_probability=min_prob, # topics with smaller probabilities are filtered out
        w_max_iter=w_max_iter, # max number of iterations to train W per each batch
        w_stop_condition=w_stop_condition, # if error difference < value, training of W stops for the current batch
        h_max_iter=h_max_iter, # max number of iterations to train h per each batch
        h_stop_condition=h_stop_condition, # if error difference < value, training of H stops for the current batch
        eval_every=eval_every, # number of batches after which l2 norm of (v - Wh) is computed (decreases performance if set too low)
        normalize=True, # normalize result
        random_state=42 # seed for random generator
    )

    # initialize Coherence Model
    cm = CoherenceModel(
        model=nmf,
        texts=texts,
        dictionary=dictionary,
        coherence='c_v'
    )

    coherence_scores.append(round(cm.get_coherence(), 5))

# pick the # of topics with highest coherence score
scores = list(zip(topic_nums, coherence_scores))
best_num_topics = sorted(scores, key=itemgetter(1), reverse=True)[0][0]
print('best_num_topics: {}'.format(best_num_topics))
idx = int(best_num_topics/5)
idx -= 1
print('coherence_score: {}'.format(coherence_scores[idx]))

# measure of word frequency in a document
# adjusted by how many documents word appears in
min_df = no_below
max_df = no_above
max_features = keep_n
tfidf_vectorizer = TfidfVectorizer(
    min_df=min_df,
    max_df=max_df,
    max_features=max_features,
    ngram_range=(1,2),
    preprocessor=' '.join
)

# returns document-term matrix (frequency of word i in document j)
tfidf = tfidf_vectorizer.fit_transform(texts)
# all the words we'll be looking at
tfidf_fn = tfidf_vectorizer.get_feature_names()

# learn a model
# sklearn model
nmf = NMF(
    n_components=best_num_topics, 
    init='nndsvd', # method for initialization procedure
    max_iter=max_iter, # max number of iterations before timing out
    l1_ratio=l1_ratio, # regularization mixing parameter (0 => l2 penalty, 1 => l1 penalty, (0,1) => mixture)
    solver='cd', # coordinate descent solver
    alpha=alpha, # constant that multiplies the regularization terms (set to 0 for no regularization)
    tol=tol, # tolerance of the stopping condition
    random_state=42 # seed for random generator
).fit(tfidf)

# transforms documents -> document-term matrix, transforms data according to model
docweights = nmf.transform(tfidf_vectorizer.transform(texts)) # (articles x topics)

# topic dataframe: (15, 8)
# (topic num : top 8 words that describe the topic)
n_top_words = 8
topic_df = topic_table(
    nmf,
    tfidf_fn,
    n_top_words
).T

# clean the topic words
topic_df['topics'] = topic_df.apply(lambda x: [' '.join(x)], axis=1)
topic_df['topics'] = topic_df['topics'].str[0]
topic_df['topics'] = topic_df['topics'].apply(lambda x: whitespace_tokenizer(x))
topic_df['topics'] = topic_df['topics'].apply(lambda x: unique_words(x))
topic_df['topics'] = topic_df['topics'].apply(lambda x: [' '.join(x)])
topic_df['topics'] = topic_df['topics'].str[0]

# clean dataframe to include topic num and topic words as a list
topic_df = topic_df['topics'].reset_index()
topic_df.columns = ['topic_num', 'topics']

topics = topic_df[['topic_num','topics']]
topics.drop_duplicates()

# merge original dataframe with topics
url = df['url'].tolist()
df_temp = pd.DataFrame({
    'url':url,
    'topic_num':docweights.argmax(axis=1)
})
print('df_temp.shape: {}'.format(df_temp.shape))
merged_topic = df_temp.merge(
    topic_df,
    on='topic_num',
    how='left'
)
print('merged_topic.shape: {}'.format(merged_topic.shape))
complete_df = merged_topic.merge(
    df,
    on='url',
    how='left'
)

complete_df = complete_df.drop('processed_text',axis=1)
complete_df.drop_duplicates()
sorted_articles = complete_df.sort_values(by=['topic_num'])
sorted_articles.to_csv(path+'sorted_articles.csv',header=True)

# get num articles per topic
num_articles_per_topic = []
for topic in range(best_num_topics):
    count = 0
    for index, row in sorted_articles.iterrows():
        if row['topic_num'] == topic:
            count += 1
    num_articles_per_topic.append(count)

topics['num_articles'] = num_articles_per_topic

# matrices from nmf (A = WH)
mat_A = tfidf_vectorizer.transform(texts)
mat_W = nmf.components_
mat_H = nmf.transform(mat_A)
# print('A = {} x {}'.format(mat_A.shape[0],mat_A.shape[1]))
# print('W = {} x {}'.format(mat_W.shape[0],mat_W.shape[1]))
# print('H = {} x {}'.format(mat_H.shape[0],mat_H.shape[1]))

# residuals: measurement of how well the topics approximate the data (observed value - predicted value)
# 0 -> topic perfectly predicts data
# residual = Frobenius norm tf-idf weights (A) - coefficients of topics (H) X coefficients of topics (W)
r = np.zeros(mat_A.shape[0]) # num articles
for row in range(mat_A.shape[0]):
    r[row] = np.linalg.norm(mat_A[row,:] - mat_H[row,:].dot(mat_W), 'fro')

sum_sqrt_res = round(sum(np.sqrt(r)),3)
print('Sum of the squared residuals is {}'.format(sum_sqrt_res))

complete_df['resid'] = r
complete_df.to_csv(path+'complete_df.csv',header=True)
resid_data = complete_df[[
    'topic_num','resid'
]].groupby('topic_num').mean().sort_values(by='resid')
complete_topics = resid_data.merge(
    topics,
    on='topic_num',
    how='left'
)
complete_topics.to_csv(path+'topics.csv',index=True,header=True)

# write parameters.txt
parameters = open(test_path+'parameters.txt','w')
parameters.write('coherence:\n')
parameters.write('\nno_below: {}'.format(no_below))
parameters.write('\nno_above: {}'.format(no_above))
parameters.write('\nkeep_n: {}'.format(keep_n))
parameters.write('\nchunksize: {}'.format(2000))
parameters.write('\npasses: {}'.format(5))
parameters.write('\nkappa: {}'.format(0.1))
parameters.write('\nminimum_probability: {}'.format(0.01))
parameters.write('\nw_max_iter: {}'.format(300))
parameters.write('\nw_stop_condition: {}'.format(0.0001))
parameters.write('\nh_max_iter: {}'.format(100))
parameters.write('\nh_stop_condition: {}'.format(0.001))
parameters.write('\neval_every: {}'.format(True))
parameters.write('\n')
parameters.write('\ntfidf_vectorizer:')
parameters.write('\nn_components: {}'.format(best_num_topics))
parameters.write('\nmin_df: {}'.format(min_df))
parameters.write('\nmax_df: {}'.format(max_df))
parameters.write('\nmax_features: {}'.format(max_features))
parameters.write('\n')
parameters.write('\nmodel:')
parameters.write('\nmax_iter: {}'.format(500))
parameters.write('\nl1_ratio: {}'.format(0.0))
parameters.write('\nalpha: {}'.format(0.0))
parameters.write('\ntol: {}'.format(1e-4))
parameters.write('\n')
parameters.write('\nsum_sqared_resids: {}'.format(sum_sqrt_res))
parameters.write('\nbest_num_topics: {}'.format(best_num_topics))
parameters.close()
