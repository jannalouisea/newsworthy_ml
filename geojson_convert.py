import pandas as pd
import numpy as np
import csv
import json
from geojson import Feature, FeatureCollection, Point
import requests
from linkpreview import link_preview
from collections import Counter


data = pd.read_csv("final_df.csv")

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
data = data[['url', 'title', 'topic', 'category', 'location', 'lat',
             'long', 'local_score', 'prov_score', 'national_score', 'inter_score']]

# header and index have to be false for the next part to work
data.to_csv("test.csv", index=False, header=False)


# convert into a proper geojson format for one article per object
# the name of the csv you read has to be the same as the one you just wrote that has no index and header and the correct columns
# the attributes in the for url, title, topic etc may require changing depending on which file you're reading
features = []
with open('test.csv', newline='') as csvfile:
    reader = csv.reader(csvfile, delimiter=',')
    for url, title, topic, category, location, latitude, longitude, local, prov, national, inter, image in reader:
        latitude, longitude = map(float, (latitude, longitude))
        features.append(
            Feature(
                geometry=Point((longitude, latitude)),
                properties={
                    'url': url,
                    'title': title,
                    'topic': topic,
                    'category': category,
                    'local_sc': local,
                    'prov_sc': prov,
                    'nat_sc': national,
                    'int_sc': inter,
                    'location': location,
                    'image': image
                }
            )
        )

# change X to indicate what you're grouping by
collection = FeatureCollection(features)
with open("grouped_by_X.json", "w") as f:
    f.write('%s' % collection)
