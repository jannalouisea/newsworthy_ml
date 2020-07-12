import newspaper
import pandas as pd
from newspaper import Article
from newspaper import fulltext

#outlet = brand

cols = ["outlet", "url", "title", "authors", "publish_date", "text", "keywords"]


# THE TORONTO STAR
star_data = []
star_news = newspaper.build("https://www.thestar.com/", memoize_articles=False)
#memoize = false because we don't want to cache articles already seen
print("TORONTO STAR...")
print(star_news.size())

#loop through all articles scraped and save into dataframe
for article in star_news.articles:
    try:
        article.download()
        article.parse()
    except newspaper.article.ArticleException:
        pass
    star_data.append([star_news.brand, article.url, article.title, article.authors, article.publish_date, article.text, article.keywords])

star = pd.DataFrame(star_data, columns=cols)
star.to_csv("tor_star.csv")


# TORONTO SUN
sun_data = []
sun_news = newspaper.build("https://torontosun.com/category/news", memoize_articles=False)
#memoize = false because we don't want to cache articles already seen
print("TORONTO SUN...")
print(sun_news.size())

#loop through all articles scraped and save into dataframe
for article in sun_news.articles:
    try:
        article.download()
        article.parse()
    except newspaper.article.ArticleException:
        pass
    sun_data.append([sun_news.brand, article.url, article.title, article.authors, article.publish_date, article.text, article.keywords])

sun = pd.DataFrame(sun_data, columns=cols)
sun.to_csv("tor_sun.csv")


#NATIONAL POST
post_data = []
post_news = newspaper.build("https://nationalpost.com/", memoize_articles=False)
#memoize = false because we don't want to cache articles already seen
print("NATIONAL POST...")
print(post_news.size())

#loop through all articles scraped and save into dataframe
for article in post_news.articles:
    try:
        article.download()
        article.parse()
    except newspaper.article.ArticleException:
        pass
    post_data.append([post_news.brand, article.url, article.title, article.authors, article.publish_date, article.text, article.keywords])

post = pd.DataFrame(post_data, columns=cols)
post.to_csv("nationalpost.csv")


#CTV 
ctv_data = []
ctv_news = newspaper.build("https://www.ctvnews.ca/", memoize_articles=False)
#memoize = false because we don't want to cache articles already seen
print("CTV NEWS...")
print(ctv_news.size())

#loop through all articles scraped and save into dataframe
for article in ctv_news.articles:
    try:
        article.download()
        article.parse()
    except newspaper.article.ArticleException:
        pass
    ctv_data.append([ctv_news.brand, article.url, article.title, article.authors, article.publish_date, article.text, article.keywords])

ctv = pd.DataFrame(ctv_data, columns=cols)
ctv.to_csv("ctv.csv")


#CBC
cbc_data = []
cbc_news = newspaper.build("https://www.cbc.ca/news", memoize_articles=False)
#memoize = false because we don't want to cache articles already seen
print("CBC NEWS...")
print(cbc_news.size())

#loop through all articles scraped and save into dataframe
for article in cbc_news.articles:
    try:
        article.download()
        article.parse()
    except newspaper.article.ArticleException:
        pass
    cbc_data.append([cbc_news.brand, article.url, article.title, article.authors, article.publish_date, article.text, article.keywords])

cbc = pd.DataFrame(cbc_data, columns=cols)
cbc.to_csv("cbc.csv")

