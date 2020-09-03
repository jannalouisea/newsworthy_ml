import guidedlda
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from gensim.corpora.dictionary import Dictionary
from gensim.models.coherencemodel import CoherenceModel
from gensim.models.ldamulticore import LdaMulticore
from operator import itemgetter
import pickle
from helper import word_count, process_text, get_unstemmed_word
from predict import predict_category


def guided_lda(filename, path):
    # parameters
    # dictionary
    no_below = 5 
    no_above = 0.25 
    keep_n = 5000 
    # count
    min_df = no_below
    max_df = no_above
    max_features = keep_n
    seed_confidence = 0.25
    # seed topics
    seed_topic_list = [['game', 'team', 'win', 'player', 'season', 'second', 'victory'],
                    ['percent', 'company', 'market', 'price', 'sell', 'business', 'stock', 'share'],
                    ['families','children','family','child','parent',]]

    seed_topic_list_processed = []
    for sub_list in seed_topic_list:
        processed_topic = []
        for word in sub_list:
            processed_topic.append(process_text(word)[0])
        seed_topic_list_processed.append(processed_topic)
    seed_topic_list = seed_topic_list_processed
    
    df = pd.read_csv(filename)
    df.drop_duplicates(subset=['text'],keep='first',inplace=True)
    df['word_count'] = df['text'].apply(word_count)
    df['processed_text'] = df['text'].apply(process_text)

    texts = df['processed_text']
    dictionary = Dictionary(texts)

    dictionary.filter_extremes(
        no_below=no_below,
        no_above=no_above,
        keep_n=keep_n
    )
    '''
    corpus = [dictionary.doc2bow(text) for text in texts]
    
    topic_nums = list(np.arange(15,55+1,5))
    coherence_scores = []
    for num in topic_nums:
        lda = LdaMulticore(
            corpus, 
            num_topics=num, 
            id2word=dictionary, 
            passes=2, 
            workers=2
        )
        cm = CoherenceModel(
            model=lda,
            texts=texts,
            dictionary=dictionary,
            coherence='c_v'
        )

        coherence_scores.append(round(cm.get_coherence(), 5))

    scores = list(zip(topic_nums, coherence_scores))
    scores = sorted(scores, key=itemgetter(1), reverse=True)
    best_num_topics = scores[0][0]
    # best_coherence_score = scores[0][1]
    
    # print(best_coherence_score)
    print(best_num_topics) # 25
    '''
    best_num_topics = 35

    count_vec = CountVectorizer(
        max_df=max_df,
        min_df=min_df,
        max_features=max_features,
        ngram_range=(1,2),
        preprocessor=' '.join
    )
    
    # print(type(texts))
    # print(texts[0])
    texts = [filter(lambda x: x in dictionary.values(), t) for t in texts]
    seed_topic_list = [filter(lambda x: x in dictionary.values(), t) for t in seed_topic_list]
    print(type(texts))
    print(type(seed_topic_list))
    
    doc_term = count_vec.fit_transform(texts) 
    vocab = count_vec.get_feature_names()
    words = dict((v, idx) for idx, v in enumerate(vocab))

    seed_topics = {}
    for t_id, st in enumerate(seed_topic_list):
        for word in st:
            try:
                seed_topics[words[word]] = t_id
            except KeyError:
                # seed word not in corpus
                print("Key Error") 
    
    alphas = [0.2]
    etas = [0.01]
    all_topics = [] # dfs
    all_models = [] # models
    all_topic_1 = [] # nums
    all_topic_2 = [] # nums
    # coherences = [] # nums
    params = {} # tuple of nums
    count = 0
    for a in alphas:
        for e in etas:
            print('alpha: {}, eta: {}'.format(a,e))

            # initialize model
            model = guidedlda.GuidedLDA(
                n_topics=best_num_topics,
                n_iter=100,
                random_state=7,
                refresh=20,
                alpha=a,
                eta=e
            )
            model.fit(doc_term, seed_topics=seed_topics, seed_confidence=seed_confidence)

            n_top_words = 10
            topic_word = model.topic_word_

            topics_list_clean = []
            topics_list_stemmed = []
            for _, topic_dist in enumerate(topic_word):
                topic_words = np.array(vocab)[np.argsort(topic_dist)][:-(n_top_words+1):-1]
                topic_words = ' '.join(topic_words)
                topic_words = list(topic_words.split(' '))
                topics_list_stemmed.append(topic_words)
                clean_topic = []
                for word in topic_words:
                    try:
                        clean_topic.append(get_unstemmed_word(word)) 
                    except KeyError:
                        clean_topic.append(word)
                        print("Key Error")
                clean_topic = ' '.join(clean_topic)
                topics_list_clean.append(clean_topic)
                # print('Topic {}: {}'.format(i, clean_topic))
            print('topic_list_stemmed: {}'.format(topics_list_stemmed))

            topics_df = pd.DataFrame(topics_list_clean,columns=['topics'])

            doc_topic = model.doc_topic_
            topics = []
            for row in doc_topic:
                new_row = np.argpartition(row, -2)[-2:]
                topics.append(new_row)
            topics = np.asarray(topics)
            topics = topics.T

            # get num articles per topic
            num_articles_per_topic_1 = []
            num_articles_per_topic_2 = []
            for topic in range(best_num_topics):
                count_topics_1 = 0
                count_topics_2 = 0
                for idx in topics[1]:
                    if topics[1][idx] == topic:
                        count_topics_1 += 1
                for idx in topics[0]:
                    if topics[0][idx] == topic:
                        count_topics_2 += 1
                num_articles_per_topic_1.append(count_topics_1)
                num_articles_per_topic_2.append(count_topics_2)

            topics_df['num_articles_1'] = num_articles_per_topic_1
            topics_df['num_articles_2'] = num_articles_per_topic_2

            '''
            # calculating coherence score
            coh_model = CoherenceModel(
                topics=topics_list_stemmed,
                texts=texts,
                dictionary=dictionary,
                coherence='c_v'
            )
            '''

            # coherences.append(round(coh_model.get_coherence(), 5))
            all_topic_1.append(topics[1]) # best 
            all_topic_2.append(topics[0]) # second best
            all_topics.append(topics_df)
            all_models.append(model)
            params[count] = (a,e)
            count += 1
    
    # params_test = np.arange(30)
    # coherence_scores = list(zip(params_test,coherences))
    # coherence_scores = sorted(coherence_scores, key=itemgetter(1))
    # best_params = coherence_scores[0][0]
    best_params = 0

    best_topics = all_topics[best_params] # df
    df['topics_1'] = all_topic_1[best_params]
    df['topics_2'] = all_topic_2[best_params]

    best_topics = predict_category(best_topics)
    df.to_csv(path+'guided_articles.csv',header=True)
    best_topics.to_csv(path+'guided_topics.csv',header=True)
    # save model, too
    # print('Best params:\na = {}, e = {}'.format(params[best_params][0], params[best_params][1]))
    

filename = 'data_subsets/no_toronto.csv'
path = '/Users/miya/Documents/GitHub/ai4good_news/news_project/guided/'
guided_lda(filename,path)
