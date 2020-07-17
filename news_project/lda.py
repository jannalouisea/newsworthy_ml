import pandas as pd
import gensim
from gensim.corpora.dictionary import Dictionary
from gensim.models.coherencemodel import CoherenceModel
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.decomposition import LatentDirichletAllocation
from gensim.models.ldamulticore import LdaMulticore
import numpy as np
from operator import itemgetter
np.random.seed(2018)
import nltk
nltk.download('wordnet')
from gensim import corpora, models

from helper import word_count, process_text, topic_table, compute_coherence_values, whitespace_tokenizer, unique_words

# parameters
# dictionary
no_below = 5 
no_above = 0.75 
keep_n = 5000 
# count
min_df = 5
max_df = 0.75
max_features = 5000
# sklearn-model lda
max_iter = 500
doc_topic_prior = 0.1
topic_word_prior = 0.2
learning_decay = 0.7
learning_offset = 50
batch_size = 128

path = '/Users/miya/Documents/GitHub/ai4good_news/news_project/test_lda_2/'
test_path = 'test_lda_2/'

df = pd.read_csv('clean.csv')
df.drop_duplicates(subset=['url'],keep='first',inplace=True)

df['word_count'] = df['text'].apply(word_count)
df['processed_text'] = df['text'].apply(process_text)

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
topic_nums = list(np.arange(5,75+1,10))

coherence_scores = []
for num in topic_nums:
    coherence_score = compute_coherence_values(bow_corpus, dictionary, num, texts)
    coherence_scores.append(coherence_score)

scores = list(zip(topic_nums, coherence_scores))
# [[topic_num_1, coherence_score_1],...]
scores = sorted(scores, key=itemgetter(1), reverse=True)
best_num_topics = scores[0][0]
best_coherence_score = scores[0][1]
print('best_num_topics: {}'.format(best_num_topics))
print('coherence_score: {}'.format(best_coherence_score))

# doc_topic_prior = 1/best_num_topics
topic_word_prior = 1/best_num_topics

max_df = max_df
min_df = min_df
max_features = max_features

# convert a collection of text documents to a matrix of token counts
tf_vectorizer = CountVectorizer(
    max_df=max_df, # max freq: num of docs containing word / total num of docs
    min_df=min_df, # min freq: word must be in at least min_df docs
    max_features=max_features, # max num of words to consider
    ngram_range=(1,2), # consider unigrams and bigrams
    preprocessor=' '.join
)
# learn the vocabulary dictionary and return document-term matrix
tf = tf_vectorizer.fit_transform(texts) 
# array mapping from feature integer indices to feature name
tf_feature_names = tf_vectorizer.get_feature_names()

lda = LatentDirichletAllocation(
    n_components=best_num_topics,
    max_iter=max_iter,
    doc_topic_prior=doc_topic_prior, # prior of document topic distribution, alpha (default=1/n_components)
    # topic_word_prior=topic_word_prior, # prior of topic word distribution, beta (default=1/n_components)
    learning_method='online',
    learning_decay=learning_decay, # learning rate (default=0.7)
    learning_offset=learning_offset, # downweights early iterations (default=10) 
    batch_size=batch_size,
    random_state=42
).fit(tf)

docweights = lda.transform(tf_vectorizer.transform(texts))

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

# coherence scores ?
score = lda.score(tf)
scores.append(score)
print('score: {}'.format(lda.score(tf)))

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
parameters.write('\ndoc_topic_prior: {}'.format(doc_topic_prior))
parameters.write('\ntopic_word_prior: {}'.format(topic_word_prior))
parameters.write('\nlearning_decay: {}'.format(learning_decay))
parameters.write('\nleanring_offset: {}'.format(learning_offset))
parameters.write('\nbatch_size: {}'.format(batch_size))
parameters.close()
