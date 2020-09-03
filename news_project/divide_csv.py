import pandas as pd

def by_date(filename, path):

    data = pd.read_csv(filename)
    data['publish_date'] = data['publish_date'].astype(str)
    data['publish_date'] = data['publish_date'].apply(lambda  x: x[:10])
    feb = data[data['publish_date'].astype(str).str.match('(.*-02-*.)')]
    march = data[data['publish_date'].astype(str).str.match('(.*-03-*.)')]
    apr = data[data['publish_date'].astype(str).str.match('(.*-04-*.)')]
    may = data[data['publish_date'].astype(str).str.match('(.*-05-*.)')]
    june = data[data['publish_date'].astype(str).str.match('(.*-06-*.)')]
    july = data[data['publish_date'].astype(str).str.match('(.*-07-*.)')]
    print(feb.shape[0]) # 18
    print(march.shape[0]) # 62 
    print(apr.shape[0]) # 52
    print(may.shape[0]) # 57
    print(june.shape[0]) # 167
    print(july.shape[0]) # 1204

    feb_march = pd.concat([feb,march])
    march_apr = pd.concat([march,apr])
    apr_may = pd.concat([apr,may])
    may_june = pd.concat([may,june])
    june_july = pd.concat([june,july])

    feb_march.to_csv(path+'feb_march_toronto.csv',header=True)
    march_apr.to_csv(path+'march_apr_toronto.csv',header=True)
    apr_may.to_csv(path+'apr_may_toronto.csv',header=True)
    may_june.to_csv(path+'may_june_toronto.csv',header=True)
    june_july.to_csv(path+'june_july_toronto.csv',header=True)

def del_toronto(filename, path):
    
    data = pd.read_csv(filename)

    data = data[data.outlet != 'torontosun']
    data = data[data.outlet != 'thestar']
    data = data[data.outlet != 'cp24']

    data.to_csv(path+'no_toronto.csv')

filename = 'clean.csv'
path = '/Users/miya/Documents/GitHub/ai4good_news/news_project/data_subsets/'
# by_date(filename,path)
by_date(filename,path)
