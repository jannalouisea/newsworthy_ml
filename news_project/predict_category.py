import pandas as pd
import pickle
from classify import get_category_name

def predict_category(filename):

    with open('tfidf.pickle', 'rb') as data:
        tfidf = pickle.load(data)

    with open('best_svc.pickle', 'rb') as data:
        model = pickle.load(data)
        
    df_topics = pd.read_csv(filename)

    categories = []
    probabilities = []
    for idx, row in df_topics.iterrows():
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

    df_topics.to_csv('/Users/miya/Documents/GitHub/ai4good_news/news_project/topics/topics_with_category.csv',header=True)