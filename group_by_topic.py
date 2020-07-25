import pandas as pd
import numpy as np
import csv
import json
from geojson import Feature, FeatureCollection, Point
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


# apply above image function to each article
data['image'] = data['url'].apply(get_image)

topic_group = pd.DataFrame()

# group articles by topic
# .apply(list) groups articles together that share a topic and lists all matching titles, urls etc in a list
topic_group['titles'] = data.groupby('topics')['title'].apply(list)
topic_group['urls'] = data.groupby('topics')['url'].apply(list)
topic_group['image_urls'] = data.groupby('topics')['image'].apply(list)
topic_group['location'] = data.groupby('topics')['location'].apply(list)
topic_group['lat'] = np.nan
topic_group['long'] = np.nan
topic_group['category'] = data.groupby('topics')['category'].apply(list)
# return only first category
topic_group['category'] = topic_group['category'].map(lambda x: x[0])


# function that returns a tuple with the most common word and the number of times it appears
def get_most_common(word_list):
    c = Counter(word_list)
    return c.most_common(1)


# returns the first 5 of a lst passed in, if the length of lst is < 5 it just returns lst
def truncate_list(lst):
    return lst[:5]


topic_group['titles'] = topic_group['titles'].apply(truncate_list)
topic_group['urls'] = topic_group['urls'].apply(truncate_list)
topic_group['image_urls'] = topic_group['image_urls'].apply(truncate_list)

# returns most common location for each topic and the count
topic_group['location'] = topic_group['location'].apply(get_most_common)
# strips the count out to just have the location
topic_group['location'] = topic_group['location'].map(lambda x: x[0][0])

# print(topic_group)

# write to csv where each data point is one topic, and includes data for the first 5 articles under that topic
topic_group.to_csv("topics_grouped.csv")
