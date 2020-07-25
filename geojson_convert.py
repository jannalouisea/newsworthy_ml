import pandas as pd
import numpy as np
import csv
import json
from geojson import Feature, FeatureCollection, Point
import requests
from linkpreview import link_preview


data = pd.read_csv("final_df.csv")

data = data[data["lat"] != 0]

# print(data)

new = data.copy()

# replace latitudes that are negative and larger than longitudes with longitude
data['lat'] = np.where(((data['lat'] < 0) & (
    np.abs(data['lat']) > np.abs(data['long']))), data['long'], data['lat'])


data['long'] = np.where((data['long'] == data['lat']),
                        new['lat'], data['long'])


data['local_score'] = np.random.randint(0, 100, size=len(data['title']))
data['prov_score'] = np.random.randint(0, 100, size=len(data['title']))
data['national_score'] = np.random.randint(0, 100, size=len(data['title']))
data['inter_score'] = np.random.randint(
    0, 100, size=len(data['title']))


data = data[['url', 'title', 'topics', 'category', 'location', 'lat',
             'long', 'local_score', 'prov_score', 'national_score', 'inter_score']]
# print(data)


# url = 'https://api.linkpreview.net?key=8c19d38dbf3c23953a76008a7b4e7f74&q=' + \
#     data['url']

# url = 'https://api.linkpreview.net?key=8c19d38dbf3c23953a76008a7b4e7f74&q=https://google.com'
# response = requests.request("POST", url)
# # print(response.text.image)
# print(response.text)
data['image'] = np.nan


def get_image(url):
    image = ""
    try:
        preview = link_preview(url)
        image = preview.image
    except requests.exceptions.HTTPError:
        pass
    return image


data['image'] = data['url'].apply(get_image)
print(data['image'])

data.to_csv("final_df_clean.csv", index=False, header=False)


# convert into a proper geojson format for one article per object
features = []
with open('final_df_clean.csv', newline='') as csvfile:
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

collection = FeatureCollection(features)
with open("articles.json", "w") as f:
    f.write('%s' % collection)
