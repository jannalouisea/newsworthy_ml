import pandas as pd
import gensim
from gensim.corpora.dictionary import Dictionary
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.decomposition import LatentDirichletAllocation
import numpy as np
from operator import itemgetter
np.random.seed(2018)
from gensim import corpora, models

from helper import topic_table, compute_coherence_values, whitespace_tokenizer, unique_words, get_qualifying_dates, get_nan_rows, make_prediction

# parameters
# dictionary
no_below = 5 
no_above = 0.50
keep_n = 5000 
# count
min_df = 5
max_df = 0.50
max_features = 5000
# sklearn-model lda
max_iter = 500
learning_offset = 30
batch_size = 128

df = get_qualifying_dates()

p_text = df['processed_text']
p_text = [item for sublist in p_text for item in sublist]
num_unique_words = len(set(p_text))

texts = df['processed_text']

dictionary = Dictionary(texts)
dictionary.filter_extremes(
    no_below=no_below, 
    no_above=no_above, 
    keep_n=keep_n
)

bow_corpus = [dictionary.doc2bow(text) for text in texts]

# find best num of topics
topic_nums = list(np.arange(15,55+1,5))
coherence_scores = []
for num in topic_nums:
    coherence_score = compute_coherence_values(
        bow_corpus, 
        dictionary, 
        num, 
        texts
    )
    coherence_scores.append(coherence_score)

scores = list(zip(topic_nums, coherence_scores))
scores = sorted(scores, key=itemgetter(1), reverse=True)
best_num_topics = scores[0][0]
best_coherence_score = scores[0][1]
print('best_num_topics: {}'.format(best_num_topics))
print('coherence_score: {}'.format(best_coherence_score))

# find best learning_decay
learning_decays = list(np.arange(0.5,1,0.1))
ld_results = []
coherence_results = []
for ld in learning_decays:
    coherence_score = compute_coherence_values(
        bow_corpus, 
        dictionary, 
        best_num_topics,
        texts
    )
    ld_results.append(ld)
    coherence_results.append(coherence_score)

model_scores = list(zip(ld_results,coherence_results)) 
model_scores = sorted(model_scores, key=itemgetter(1), reverse=True)
best_coherence_score = model_scores[0][1]
learning_decay = model_scores[0][0]
print('best_learning_decay: {}'.format(learning_decay))
print('coherence_score: {}'.format(best_coherence_score))

# convert a collection of text documents to a matrix of token counts
'''
tf_vectorizer = TfidfVectorizer(
    max_df=max_df, # max freq: num of docs containing word / total num of docs
    min_df=min_df, # min freq: word must be in at least min_df docs
    max_features=max_features, # max num of words to consider
    ngram_range=(1,2), # consider unigrams and bigrams
    preprocessor=' '.join
)'''
tf_vectorizer = CountVectorizer(
    max_df=max_df,
    min_df=min_df,
    max_features=max_features,
    ngram_range=(1,2),
    preprocessor=' '.join
)

# learn the vocabulary dictionary and return document-term matrix
tf = tf_vectorizer.fit_transform(texts) 
# array mapping from feature integer indices to feature name
tf_feature_names = tf_vectorizer.get_feature_names()
# document-topic matrix
tf_transformed = tf_vectorizer.transform(texts)

path = '/Users/miya/Documents/GitHub/ai4good_news/news_project/test_lda/'
test_path = 'test_lda/'

lda = LatentDirichletAllocation(
    n_components=best_num_topics,
    max_iter=max_iter,
    learning_method='online',
    learning_decay=learning_decay, 
    learning_offset=learning_offset, # downweights early iterations (default=10) 
    batch_size=batch_size,
    random_state=42
).fit(tf)

docweights = lda.transform(tf_transformed)

n_top_words = 8
topic_df = topic_table(
    lda,
    tf_feature_names,
    n_top_words
).T

topic_df['topics'] = topic_df.apply(lambda x: [' '.join(x)], axis=1)
topic_df['topics'] = topic_df['topics'].str[0]
topic_df['topics'] = topic_df['topics'].apply(lambda x: whitespace_tokenizer(x))
topic_df['topics'] = topic_df['topics'].apply(lambda x: unique_words(x))
topic_df['topics'] = topic_df['topics'].apply(lambda x: [' '.join(x)])
topic_df['topics'] = topic_df['topics'].str[0]

topic_df = topic_df['topics'].reset_index()
topic_df.columns = ['topic_num', 'topics']

topics = topic_df[['topic_num','topics']]
topics.drop_duplicates()

url = df['url'].tolist()
df_temp = pd.DataFrame({
    'url':url,
    'topic_num':docweights.argmax(axis=1)
})
merged_topic = df_temp.merge(
    topic_df,
    on='topic_num',
    how='left'
)
complete_df = pd.merge(
    df,
    merged_topic,
    on='url',
    how='left'
)
complete_df = complete_df.drop('processed_text',axis=1)
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
topics.to_csv(path+'topics.csv',index=True,header=True)

perplexity = lda.perplexity(tf)

# write parameters.txt
parameters = open(test_path+'parameters.txt','w')
parameters.write('coherence:')
parameters.write('\nno_below: {}'.format(no_below))
parameters.write('\nno_above: {}'.format(no_above))
parameters.write('\nkeep_n: {}'.format(keep_n))
parameters.write('\n')
parameters.write('\ntfidf_vectorizer:')
parameters.write('\nn_components: {}'.format(best_num_topics))
parameters.write('\nmax_iter: {}'.format(max_iter))
parameters.write('\nlearning_decay: {}'.format(learning_decay))
parameters.write('\nleanring_offset: {}'.format(learning_offset))
parameters.write('\nbatch_size: {}'.format(batch_size))
parameters.write('\n')
parameters.write('\nbest_num_topics: {}'.format(best_num_topics))
parameters.write('\ncoherence_score: {}'.format(best_coherence_score))
parameters.write('\nperplexity: {}'.format(perplexity))
parameters.close()

# perplex = open(test_path+'perplexities.txt','w')
# perplex.write('perplexities:\n')
# for p in perplexities:
#     perplex.write(p+', ')
# perplex.write('\nparameters:\n')
# for p in params:
#     perplex.write(str(p)+' : '+str(params[p]))
# perplex.write('test #{} had best perplexity score\n'.format(best_params))
# perplex.write('ld={} had best coherence score'.format(learning_decay))
# perplex.write('params: ld={}'.format(params[best_params]))
# perplex.close()

print('params: ld={}'.format(learning_decay))
print('coherence score: {}'.format(best_coherence_score))
print('perplexity scores: {}'.format(perplexity))

# make prediction on unseen articles
df_predictions = make_prediction(lda,tf_vectorizer)
df_predictions.to_csv('/Users/miya/Documents/GitHub/ai4good_news/news_project/test_lda/unseen_articles.csv',header=True)
