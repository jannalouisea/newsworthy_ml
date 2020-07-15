import requests
from bs4 import BeautifulSoup
import numpy as np
import pandas as pd
import time


# # # # # #  THE RECORD NEWS # # # # # # # # #

# LOCAL
url = 'https://www.therecord.com/business.html'

response = requests.get(url, timeout=5)
local = response.content
soup = BeautifulSoup(local, 'html.parser')
local_news = soup.find_all('a', class_='c-mediacard c-article-list-flex__article c-mediacard--row-reverse', href=True)
local_news2 = soup.find_all('a', class_='c-mediacard c-article-list-flex__article c-mediacard--column', href=True)

local_news = local_news + local_news2
num_art = len(local_news)
print(num_art)

authors = []
content = []
links = []
titles = []
pub_dates = []


for n in np.arange(0, num_art):
    # Storing article links
    link = 'https://www.therecord.com' + local_news[n]['href']
    links.append(link)

    response = requests.get(link)
    art_content = response.content
    article = BeautifulSoup(art_content, 'html.parser')

    title = article.find('h1', class_='c-article-headline__heading').get_text()
    print(title)
    titles.append(title)

    # Storing the authors
    author = article.find('div', class_='article__byline')
    art_authors = []
    for span in author.find_all('span', recursive=False):
        auth = span.get_text()
        if auth in ['By ']:
            continue
        auth = auth.replace('Record Reporter', '')
        auth = auth.replace('Local Journalism', '')
        auth = auth.replace('Cambridge Times', '')
        auth = auth.replace('Waterloo Chronicle', '')
        auth = auth.replace('Special to the Record', '')
        auth = auth.replace('Record Columnist', '')
        auth = auth.replace('Initiative Reporter', '')
        art_authors.append(auth)
    authors.append(art_authors)

    # Storing publish date
    pub_date = article.find('div', class_='article__time-container')

    for span in pub_date.span.find_all('span', recursive=False):
        date = span.get_text()
        if date=='':
            continue
        pub_dates.append(date)
        print(date)

    # Storing article content
    body = article.find_all('div', class_='c-article-body__content')
    x = body[0].find_all('p')

    paras = []          # Unifying the paragraphs
    for p in np.arange(0, len(x)):
        paragraph = x[p].get_text()
        paras.append(paragraph)
        final_article = " ".join(paras)

    content.append(final_article)

print(len(authors))
print(len(content))
print(len(links))
print(len(titles))
print(len(pub_dates))

print(pub_dates)

# Storing in a dataframe
local_df = pd.DataFrame({'outlet': 'The Record', 'url': links, 'title': titles, 'authors': authors, 'publish_date': pub_dates, 'text': content})
local_df.to_csv('the_record_business.csv')