import pandas as pd
import numpy as np
import csv
import json
# from geojson import Feature, FeatureCollection, Point
import requests
from linkpreview import link_preview
from collections import Counter

data = pd.read_csv("final_df.csv")

# add image column to data
data['image'] = np.nan

# use api to get image for each article, or use default if none exists
def get_image(url):
    image = "https://sisterhoodofstyle.com/wp-content/uploads/2018/02/no-image-1.jpg"
    try:
        preview = link_preview(url)
        image = preview.image
    except requests.exceptions.HTTPError:
        pass
    return image

def truncate_list(lst):
    return lst[:5]

# data['image'] = data['url'].apply(get_image)

topic_nums = data.topic_num.unique()
topic_groups = data.groupby('topic_num')
topic_loc_df = pd.DataFrame()
for num in topic_nums:
    topic_df = topic_groups.get_group(num)
    location_groups = pd.DataFrame()
    location_groups['topic'] = topic_df.groupby('location')['topics'].apply(list)
    location_groups['topic'] = location_groups['topic'].map(lambda x: x[0])
    location_groups['topic_num'] = topic_df.groupby('location')['topic_num'].apply(list)
    location_groups['topic_num'] = location_groups['topic_num'].map(lambda x: x[0])
    location_groups['titles'] = topic_df.groupby('location')['title'].apply(list)
    location_groups['urls'] = topic_df.groupby('location')['url'].apply(list)
    # location_groups['image_urls'] = topic_df.groupby('location')['image'].apply(list)
    location_groups['lat'] = topic_df.groupby('location')['lat'].apply(list)
    location_groups['lat'] = location_groups['lat'].map(lambda x: x[0])
    location_groups['long'] = topic_df.groupby('location')['long'].apply(list)
    location_groups['long'] = location_groups['long'].map(lambda x: x[0])
    location_groups['category'] = topic_df.groupby('location')['category'].apply(list)
    location_groups['category'] = location_groups['category'].map(lambda x: x[0])
    
    topic_loc_df = topic_loc_df.append(location_groups)

topic_loc_df['titles'] = topic_loc_df['titles'].apply(truncate_list)
topic_loc_df['urls'] = topic_loc_df['urls'].apply(truncate_list)
# topic_loc_df['images_urls'] = topic_loc_df['image_urls'].apply(truncate_list)

print(topic_loc_df)

urls = []
for idx,row in topic_loc_df.iterrows():
    url_list = row['urls']
    image_list = []
    for url in url_list:
        image_list.append(get_image(url))
    urls.append(url_list)

topic_loc_df['image_urls'] = urls
    

topic_loc_df.to_csv('/Users/miya/Documents/GitHub/ai4good_news/news_project/topic_location.csv',index=False,header=True)