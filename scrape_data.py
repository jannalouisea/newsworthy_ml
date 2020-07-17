import newspaper
import pandas as pd
from newspaper import Article
from newspaper import fulltext

#outlet = brand

cols = ["outlet", "url", "title", "authors", "publish_date", "text", "keywords"]

urls = {
    "cbc" : "https://www.cbc.ca/news",
    "ctv" : "https://www.ctvnews.ca/",
    "national_post" : "https://nationalpost.com/",
    "toronto_sun" : "https://torontosun.com/category/news",
    "toronto_star" : "https://www.thestar.com/",
    "cp_24" : "https://www.cp24.com/",
    "maple_ridge" : "https://www.mapleridgenews.com/local-news/",
    "tri_city" : "https://www.tricitynews.com/",
    "langley_advance_times" : "https://www.langleyadvancetimes.com/local-news/",
    "abbotsford" : "https://www.abbynews.com/local-news/",
    "chilliwack_progress" : "https://www.theprogress.com/local-news/",
    "delta_optimist" : "https://www.delta-optimist.com/local-news",
    "north_delta_reporter" : "https://www.northdeltareporter.com/local-news/",
    "surrey_now_leader" : "https://www.surreynowleader.com/local-news/",
    "vancouver_observer" : "https://www.vancouverobserver.com/",
    "vancouver_courier" : "https://www.vancourier.com/",
    "georgia_straight" : "https://www.straight.com/news",
    "north_shore" : "https://www.nsnews.com/",
    "richmond" : "https://www.richmond-news.com/",
    "richmond_senitel" : "https://richmondsentinel.ca/",
    "burnaby_now" : "https://www.burnabynow.com/",
    "new_west_record" : "https://www.newwestrecord.ca/",
    "bowen_island_undercurrent" : "https://www.bowenislandundercurrent.com/"
}

# aggregated = pd.DataFrame(columns=cols)
run = 0
for url in urls:
    data = []
    run = run + 1
    news = newspaper.build(urls[url], memoize_articles=False)
    #memoize = false because we don't want to cache articles already seen

    print(url + " news...")
    print(news.size())

    #loop through all articles scraped and save into dataframe for each news outlet
    for article in news.articles:
        try:
            article.download()
            article.parse()
        except newspaper.article.ArticleException:
            pass
        # append each article and its fields into list
        data.append([news.brand, article.url, article.title, article.authors, article.publish_date, article.text, article.keywords])

    # append each news outlets data into the csv as we go
    chunk = pd.DataFrame(data, columns=cols)
    if run == 1:
        chunk.to_csv("all_news.csv")
    else:
        chunk.to_csv("all_news.csv", header=None, mode="a")





