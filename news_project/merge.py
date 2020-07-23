import pandas as pd

df_topics = pd.read_csv('topics/topics_with_category_0.csv')
df_articles = pd.read_csv('test_nmf/0/sorted_articles.csv')

url = df_articles['url']
topic_num = df_articles['topic_num']
df_temp = pd.DataFrame({
    'url':url,
    'topic_num':topic_num,
})
merged = df_temp.merge(
    df_topics,
    on='topic_num',
    how='left'
)
complete = merged.merge(
    df_articles,
    on='url',
    how='left'
)
complete = complete.drop(columns=['topic_num_x','topics_x','Unnamed: 0_x','Unnamed: 0.1_x','Unnamed: 0_y','Unnamed: 0.1_y'])
complete = complete.rename(columns={'topic_num_y':'topic_num','topics_y':'topics','num_articles':'num_articles_per_topic','resid_x':'topic_resid','resid_y':'article_resid','probability':'category_probability'})


print(complete.head())
complete.to_csv('/Users/miya/Documents/GitHub/ai4good_news/news_project/complete.csv',header=True)