import pandas as pd
import pickle
import numpy as np
from random import seed
from random import random
from helper import get_category_name, process_text

def predict_category(df):

    with open('svc_tfidf_trimmed_v13.pickle', 'rb') as data:
        tfidf = pickle.load(data)

    with open('svc_model_trimmed_v13.pickle', 'rb') as data:
        model = pickle.load(data)

    # s3 = boto3.client('s3')

	# model_obj = s3.get_object(Bucket='sagemaker-studio-i7gmskjysd', Key='svc_model.pickle')
	# serialized_model = model_obj['Body'].read()
	# model = pickle.loads(serialized_model)

    # tfidf_obj = s3.get_object(Bucket='sagemaker-studio-i7gmskjysd', Key='svc_tfidf.pickle')
    # serialized_tfidf = tfidf_obj['Body'].read()
    # tfidf = pickle.loads(serialized_tfidf)
        
    df_topics = df

    categories = []
    probabilities = []
    for _, row in df_topics.iterrows():
        topic_matrix = tfidf.transform([row['topics']])
        topic_matrix = topic_matrix.toarray()
        svc_pred = model.predict(topic_matrix)[0]
        svc_pred_proba = model.predict_proba(topic_matrix)[0]
        category = get_category_name(svc_pred)
        probability = svc_pred_proba.max()*100
        categories.append(category)
        probabilities.append(probability)
        
    df_topics['category'] = categories
    df_topics['probability'] = probabilities

    return df_topics


def predict_topic(filename):

    with open('nmf_model.pickle', 'rb') as data:
        model = pickle.load(data)

    with open('nmf_tfidf.pickle', 'rb') as data:
        tfidf = pickle.load(data)

    # s3 = boto3.client('s3')

    # model_obj = s3.get_object(Bucket='sagemaker-studio-i7gmskjysd', Key='nmf_model.pickle')
    # serialized_model = model_obj['Body'].read()
    # model = pickle.loads(serialized_model)

    # tfidf_obj = s3.get_object(Bucket='sagemaker-studio-i7gmskjysd', Key='nmf_tfidf.pickle')
    # serialized_tfidf = tfidf_obj['Body'].read()
    # tfidf = pickle.loads(serialized_tfidf)
    
    df = pd.read_csv(filename)
    # role = get_execution_role()
    # bucket='sagemaker-studio-i7gmskjysd'
    # data_key = filename
    # data_location = 's3://{}/{}'.format(bucket, data_key)
    # df = pd.read_csv(data_location)

    # process text
    df['processed_text'] = df['text'].apply(process_text)
    new_texts = df['processed_text']

    # transform data with fitted models
    tfidf_unseen = tfidf.transform(new_texts)
    X_new = model.transform(tfidf_unseen)

    # top predicted topics
    predicted_topics = [np.argsort(each)[::-1][0] for each in X_new]
    df['pred_topic_num'] = predicted_topics

    return df
    # bucket='sagemaker-studio-i7gmskjysd'
    # data_key = filename
    # data_location = 's3://{}/{}'.format(bucket, data_key)
    # df.to_csv(data_location, index=False)

def count_categories(df):

    seed(1)
    rand_col = []
    for i in range(df.shape[0]):
        rand_col.append(random())
    
    df['seed'] = rand_col
    print('before: '+str(df.shape))
    # 121
    # df = df[(df['category'] != 0) | (df['seed'] <= 0.571)] # business - even: 0.575
    # df = df[(df['category'] != 1) | (df['seed'] <= 0.63)] # arts & entertainment - even: 0.64
    # df = df[(df['category'] != 2) | (df['seed'] <= 0.568)] # politics - even: 0.5707
    # df = df[(df['category'] != 3) | (df['seed'] <= 0.575)] # sports
    # df = df[(df['category'] != 5) | (df['seed'] <= 0.41)] # covid - even: 0.43 
    # df = df[(df['category'] != 6) | (df['seed'] <= 0.4245)] # lifestyle - even: 0.4265 ; 101: 0.424
    # df = df[(df['category'] != 7) | (df['seed'] <= 0.3253)] # local - even: 0.326 ; 101: 0.323
    df = df[(df['category'] != 0) | (df['seed'] <= 0.8346)] # business 
    df = df[(df['category'] != 1) | (df['seed'] <= 0.967)] # arts & entertainment 
    df = df[(df['category'] != 2) | (df['seed'] <= 0.8195)] # politics 
    df = df[(df['category'] != 3) | (df['seed'] <= 0.95)] # sports
    df = df[(df['category'] != 5) | (df['seed'] <= 0.74)] # covid  
    df = df[(df['category'] != 6) | (df['seed'] <= 0.5893)] # lifestyle 
    df = df[(df['category'] != 7) | (df['seed'] <= 0.505)] # local 
    print('after: '+str(df.shape))

    counts = []
    for cat in range(10):
        count = 0
        for _, row in df.iterrows():
            if(row['category'] == cat):
                count += 1
        counts.append(count)
    
    for i in range(10):
        print('{}: {}'.format(get_category_name(i), counts[i]))
    print('total: {}'.format(sum(counts)))

    df.to_csv('/Users/miya/Documents/GitHub/ai4good_news/news_project/all_articles_trimmed.csv',header=True)

# df = predict_category('test_nmf/complete_topics.csv')
# df.to_csv('topics_with_categories.csv')
'''
topics = ['films music story bowen songs island books world', # a&e
            'cases health tested new province virus deaths outbreaks', # covid 
            'players teams game season league play hockey sports', # sports
            'police rcmp investigate man charge office incidents crime', # local
            'burnaby photos new west photograph jennifer gauthier',  # local (lifestyle?)
            'school students programming learn educator children parents kids', # lifestyle (politics?)
            'government canadians federal businesses billion trudeau minister million', # politics
            'north shores news vancouver photos',  # local
            'patients dr dental clinic client\'s care health denture', # health
            'funeral home forest lawn arrange knapman', # local
            'restaurants food custom businesses store dishes local menu', # local (lifestyle?) 
            'city parking council houses building street site resident', # local
            'arts artists gallery exhibit painting culture council studio', # a&e (local?)
            'mask wear face public cover mandatory', # covid
            'car vehicle engineers prices driving driver model wheels', # business
            'science tech AI cell microscope computer space', # scitech
            'house market stock price sell real estate'] # business

labels = ['arts & entertainment', 'covid-19', 'sports', 'local', 'local', 
            'local', 'politics', 'local', 'health', 'local', 'local', 
            'local', 'arts & entertainment', 'covid-19', 'business', 
            'science & tech', 'business']
'''

topics = ['life loving don friends world ve something loads', 
            'th login username subscriber news office contact hi sorry order viewed', 
            'school students district parent learning class teacher plan', 
            'health case testing hospital careful outbreak deaths province', 
            'game team plays seasons players league hockey coaching', 
            'investments td investors financial tax advisor banked insurance', 
            'party conservation tools election government liberals mr trudeau', 
            'kamloops donation week support local afford free media advertising', 
            'lake parking trailing wildfire bc city rivers area', 
            'estate real house prices market gallery agent', 
            'police rcmp man investigation office court reported black', 
            'art festivities film artist music event theatre performance', 
            'company fish product seafood prices market salmon million',
            'burnaby business foods photograph photos jennifer restaurant customers', 
            'vehicle car drive driver engineering wheel model litre'
]

# data = {'topics': topics, 'labels':labels}
# print(len(topics))
# print(len(labels))
# df = pd.DataFrame(data=topics,columns=['topics'])

# data = pd.read_csv('all_articles.csv')
# count_categories(data)
# print(predict_category(df))



