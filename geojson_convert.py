import pandas as pd
import numpy as np
import csv
import json
from geojson import Feature, FeatureCollection, Point
import requests
from linkpreview import link_preview
from collections import Counter


data = pd.read_csv("topic_location.csv")
# drop articles with no location
data = data[data["lat"] != 0]

# NOTE: only uncomment this section if some of the latitudes and longitudes are mixed up
# latitude should be positive, longitude should be negative (in most cases for BC locations)
# new = data.copy()

# replace latitudes that are negative and larger than longitudes with longitude
# data['lat'] = np.where(((data['lat'] < 0) & (
#     np.abs(data['lat']) > np.abs(data['long']))), data['long'], data['lat'])


# data['long'] = np.where((data['long'] == data['lat']),
#                         new['lat'], data['long'])


# NOTE: to assign random scores for relevancy comment out the following lines
# data['local_score'] = np.random.randint(0, 100, size=len(data['title']))
# data['prov_score'] = np.random.randint(0, 100, size=len(data['title']))
# data['national_score'] = np.random.randint(0, 100, size=len(data['title']))
# data['inter_score'] = np.random.randint(
#     0, 100, size=len(data['title']))


# only keep the columns you want -> may require changing
data = data[['urls', 'titles', 'topic', 'category', 'location', 'lat',
             'long', 'bc_score', 'canada_score', 'world_score', 'image_urls']]

for idx,row in data.iterrows():
    if row['bc_score'] == '<1':
        data.at[idx,'bc_score'] = str(1)
    if row['canada_score'] == '<1':
        data.at[idx,'canada_score'] = str(1)
    if row['world_score'] == '<1':
        data.at[idx,'world_score'] = str(1)

data['bc_score'] = data['bc_score'].astype(float)
data['canada_score'] = data['canada_score'].astype(float)
data['world_score'] = data['world_score'].astype(float)

# header and index have to be false for the next part to work
data.to_csv("test.csv", index=False, header=False)


# convert into a proper geojson format for one article per object
# the name of the csv you read has to be the same as the one you just wrote that has no index and header and the correct columns
# the attributes in the for url, title, topic etc may require changing depending on which file you're reading
features = []
with open('test.csv', newline='') as csvfile:
    reader = csv.reader(csvfile, delimiter=',')
    for url, title, topic, category, location, latitude, longitude, prov, national, inter, image in reader:
        latitude, longitude = map(float, (latitude, longitude))
        features.append(
            Feature(
                geometry=Point((longitude, latitude)),
                properties={
                    'url': url,
                    'title': title,
                    'topic': topic,
                    'category': category,
                    'prov_sc': float(prov),
                    'nat_sc': float(national),
                    'int_sc': float(inter),
                    'location': location,
                    'image': image
                }
            )
        )

# change X to indicate what you're grouping by
collection = FeatureCollection(features)
with open("grouped_by_X_2.json", "w") as f:
    f.write('%s' % collection)
