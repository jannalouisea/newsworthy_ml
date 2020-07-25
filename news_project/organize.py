import pandas as pd
import os

# articles with location + topics with score
df = pd.read_csv('final_df.csv')
num_topics = df['topic_num'].nunique()
grouped = df.groupby(df.topic_num)

for topic in range(num_topics):
    df_topic = grouped.get_group(topic)
    locations = df_topic.location.unique()
    num_locs = len(locations)
    topic_locs = df_topic.groupby(df_topic.location)
    print('topic: ',topic)
    # parent_path = '/Users/miya/Documents/GitHub/ai4good_news/news_project/groups'
    # dir_name = str(topic)
    # path = os.path.join(parent_path, dir_name) 
    # os.mkdir(path) 
    for loc in locations:
        df_group = topic_locs.get_group(loc)
        loc = loc.replace('.','')
        loc = loc.replace('/','_')
        if df_group.shape[0] > 2:
            print(df_group.shape[0])
            file_loc = '/Users/miya/Documents/GitHub/ai4good_news/news_project/groups/{}/{}.csv'.format(topic,loc)
            df_group.to_csv(file_loc,index=False,header=True)