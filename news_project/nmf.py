# !pip3 install nbimporter
# !pip3 install nbformat
# !pip3 install sagemaker
# !pip3 install s3fs

import pandas as pd
import numpy as np
from sklearn.decomposition import NMF
from sklearn.feature_extraction.text import TfidfVectorizer
from gensim.models.coherencemodel import CoherenceModel
from gensim.corpora.dictionary import Dictionary
from gensim.models.nmf import Nmf
from operator import itemgetter
import pickle
import sys
import os

from helper import word_count, process_text, topic_table, whitespace_tokenizer, unique_words, get_unstemmed_word
from predict import predict_category

# to change for sagemaker:
#   uncomment install statements
#   change import statements for functions from helper, predict
#   change csv and model reading/writing and dumping/loading to s3 versions
#   **may have to change bucket name
def nmf(filename1, path):
    # parameters
    # dictionary
    no_below = 5 
    no_above = 0.25 
    keep_n = 3000 
    # gensim-coherence nmf
    chunksize = 1000
    passes = 10
    kappa = 0.1
    min_prob = 0.001
    w_max_iter = 300
    w_stop_condition = 0.0001
    h_max_iter = 100
    h_stop_condition = 0.001
    eval_every = True
    # tf-idf
    min_df = no_below
    max_df = no_above
    max_features = keep_n
    # sklearn-model nmf
    max_iter = 500
    tol = 0.0001

    # get data
    df = pd.read_csv(filename1)
    # df2 = pd.read_csv(filename2)
    # df3 = pd.read_csv(filename3)
    # df = pd.concat([df,df2,df3])
    
    # role = get_execution_role()
    # bucket='sagemaker-studio-i7gmskjysd'
    # data_key = filename
    # data_location = 's3://{}/{}'.format(bucket, data_key)
    # df = pd.read_csv(data_location)

    df.drop_duplicates(subset=['text'],keep='first',inplace=True) 
    df = df[df['text'].notna()]
    
    # process text
    df['word_count'] = df['text'].apply(word_count)
    df['processed_text'] = df['text'].apply(process_text)

    # create dictionary to pass as input to gensim model
    texts = df['processed_text']
    dictionary = Dictionary(texts)

    # filter out words 
    dictionary.filter_extremes(
        no_below=no_below,
        no_above=no_above,
        keep_n=keep_n
    )
    
    # convert to bag of words (corpus) to pass to gensim nmf model
    # [[(word_id, # times word appears in document),...],...]
    corpus = [dictionary.doc2bow(text) for text in texts]
    
    # find optimal # of topics
    topic_nums = list(np.arange(15,55+1,5))
    coherence_scores = []
    for num in topic_nums:
        # initialize NMF model
        nmf = Nmf(
            corpus=corpus,
            num_topics=num,
            id2word=dictionary, 
            chunksize=chunksize, 
            passes=passes, 
            kappa=kappa, 
            minimum_probability=min_prob, 
            w_max_iter=w_max_iter, 
            w_stop_condition=w_stop_condition, 
            h_max_iter=h_max_iter, 
            h_stop_condition=h_stop_condition, 
            eval_every=eval_every, 
            normalize=True, 
            random_state=42
        )

        # initialize Coherence Model
        cm = CoherenceModel(
            model=nmf,
            texts=texts,
            dictionary=dictionary,
            coherence='c_v'
        )

        coherence_scores.append(round(cm.get_coherence(), 5))

    scores = list(zip(topic_nums, coherence_scores))
    scores = sorted(scores, key=itemgetter(1), reverse=True)
    best_num_topics = scores[0][0]
    # print(best_num_topics)
    
    # measure of word frequency in a document (adjusted)
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

    # fit+transform: returns document-term matrix (frequency of word i in document j)
    tfidf = tfidf_vectorizer.fit_transform(texts)
    # all the words we'll be looking at
    tfidf_fn = tfidf_vectorizer.get_feature_names()
    
    # learn a model
    nmf = NMF(
        n_components=best_num_topics, 
        init='nndsvd', 
        max_iter=max_iter, 
        l1_ratio=0.0, 
        solver='cd', 
        alpha=0.0, 
        tol=tol, 
        random_state=42 
    ).fit(tfidf)

    try:
        # transforms documents -> document-term matrix, transforms data according to model
        docweights = nmf.transform(tfidf) # (articles x topics)

        # topic dataframe: (best_num_topics x 8)
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

        # clean topic dataframe 
        topic_df = topic_df['topics'].reset_index()
        topic_df.columns = ['topic_1', 'topics']

        topics = topic_df[['topic_1','topics']]

        # find top two topics for each document with docweights array
        topics_list = []
        for row in docweights:
            new_row = np.argpartition(row, -2)[-2:]
            topics_list.append(new_row)
        topics_list = np.asarray(topics_list)
        topics_list = topics_list.T

        # assign topics to each article
        # topic_list[1]: most likely topic
        # topic_list[0]: 2nd most likley topic
        url = df['url'].tolist()
        df_temp = pd.DataFrame({
            'url':url,
            'topic_1':topics_list[1],
            'topic_2':topics_list[0]
        })
        merged_topic = df_temp.merge(
            topic_df,
            on='topic_1',
            how='left'
        )
        complete_df = merged_topic.merge(
            df,
            on='url',
            how='left'
        )

        complete_df = complete_df.drop('processed_text',axis=1)
        complete_df = complete_df.drop_duplicates()
        sorted_articles = complete_df.sort_values(by=['topic_1'])
        
        # get num articles per topic (top two)
        num_articles_per_topic_1 = []
        num_articles_per_topic_2 = []
        for topic in range(best_num_topics):
            count_topics_1 = 0
            count_topics_2 = 0
            for idx_1 in topics_list[1]:
                if idx_1 == topic:
                    count_topics_1 += 1
            for idx_2 in topics_list[0]:
                if idx_2  == topic:
                    count_topics_2 += 1
            num_articles_per_topic_1.append(count_topics_1)
            num_articles_per_topic_2.append(count_topics_2)

        topics['num_articles_1'] = num_articles_per_topic_1
        topics['num_articles_2'] = num_articles_per_topic_2
        
        # matrices from nmf (A = WH)
        mat_A = tfidf_vectorizer.transform(texts)
        mat_W = nmf.components_
        mat_H = nmf.transform(mat_A)

        # residuals: measurement of how well the topics approximate the data (observed value - predicted value)
        # 0 -> topic perfectly predicts data
        # residual = Frobenius norm tf-idf weights (A) - coefficients of topics (H) X coefficients of topics (W)
        r = np.zeros(mat_A.shape[0]) # num articles
        for row in range(mat_A.shape[0]):
            r[row] = np.linalg.norm(mat_A[row,:] - mat_H[row,:].dot(mat_W), 'fro')

        # add avg residual column to topics
        complete_df['resid'] = r
        sorted_articles = complete_df.sort_values(by=['topic_1'])
        resid_data = complete_df[[
            'topic_1','resid'
        ]].groupby('topic_1').mean().sort_values(by='resid')
        complete_topics = topics.merge(
            resid_data,
            on='topic_1',
            how='left'
        )

    except Exception:
        exc_type, _, exc_tb = sys.exc_info()
        fname = os.path.split(exc_tb.tb_frame.f_code.co_filename)[1]
        print(exc_type, fname, exc_tb.tb_lineno)

    complete_topics = predict_category(complete_topics)
    complete_topics.to_csv(path+'complete_topics_unstemmed_temp.csv',header=True)

    # unstem topic words
    for idx, row in complete_topics.iterrows():
        new_words = ''
        topics_itr = row['topics'].split()
        for word in topics_itr:
            try:
                new_words += get_unstemmed_word(word) 
            except KeyError:
                print('Key Error: {}'.format(word))
                new_words += word
            new_words += ' '
        complete_topics.at[idx, 'topics'] = new_words

    categories = []
    for idx, row in sorted_articles.iterrows():
        topic_num = row['topic_1']
        topics = complete_topics.at[topic_num, 'topics']
        categories.append(complete_topics.at[topic_num,'category'])
        sorted_articles.at[idx, 'topics'] = topics
    sorted_articles['category'] = categories

    sorted_articles = sorted_articles.drop('Unnamed: 0',axis=1)

    # temporary bubble size: number of articles per topic
    bubble_sizes = []
    for _, row in sorted_articles.iterrows():
        bubble_sizes.append(complete_topics.at[row['topic_1'], 'num_articles_1'])
    sorted_articles['bubble_size'] = bubble_sizes

    # sorted_articles.to_csv(path+'sorted_articles_new_data.csv', header=True)
    complete_topics.to_csv(path+'complete_topics_temp.csv',header=True)

    # data_key_articles = 'sorted_articles.csv'
    # data_location_articles = 's3://{}/{}'.format(bucket, data_key_articles)
    # best_articles.to_csv(data_location_articles,header=True,index=False)
    # data_key_topics = 'complete_topics.csv'
    # data_location_topics = 's3://{}/{}'.format(bucket, data_key_topics)
    # best_topics.to_csv(data_location_topics,header=True,index=False)

    # save model
    with open('nmf_model.pickle', 'wb') as output:
        pickle.dump(nmf, output)

    with open('nmf_tfidf.pickle', 'wb') as output:
        pickle.dump(tfidf_vectorizer, output)

    # s3 = boto3.client('s3')
    
    # serialized_model = pickle.dumps(models[best_params])
    # s3.put_object(Bucket='sagemaker-studio-i7gmskjysd', Key='nmf_model.pickle', Body=serialized_model)
    
    # serialized_tfidf = pickle.dumps(tfidf_vectorizer)
    # s3.put_object(Bucket='sagemaker-studio-i7gmskjysd', Key='nmf_tfidf.pickle', Body=serialized_tfidf)
    
bc_dailies = 'scraped_articles/BC_Dailies.csv'
# national = 'scraped_articles/National.csv'
# local_weeklies = 'scraped_articles/Local_Weeklies.csv'
path = '/Users/miya/Documents/GitHub/ai4good_news/news_project/nmf/'

nmf(bc_dailies,path)