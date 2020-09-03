import pandas as pd
from predict import predict_category
from helper import process_text, get_unstemmed_word, casual_tokenizer

# df = pd.read_csv('sorted_articles_with_category.csv')

# articles from each news outlet
# articels from each category

# outlets = df.outlet.unique().tolist()
# categories = ['lifestyle','covid-19','sports','local','politics','business','arts & entertainment']

# print(df.shape)
# count_outlets = {}
# for o in outlets:
#     count_outlets[o] = 0

# count_cats = {}
# for c in categories:
#     count_cats[c] = 0

# for _, row in df.iterrows():
#     for o in outlets:
#         if row['outlet'] == o:
#             count_outlets[o] += 1
#     for c in categories:
#         if row['category'] == c:
#             count_cats[c] += 1

# print('categories: ',count_cats)
# print('outlets: ',count_outlets)

# count_cats = {}
# for c in categories:
#     count_cats[c] = 0

# df = df.drop(['category'], axis=1)
# df = predict_category(df)
# for _, row in df.iterrows():
#     for c in categories:
#         if row['category'] == c:
#             count_cats[c] += 1
    
# print('categories: ',count_cats)
# for idx, o in enumerate(outlets):
#     print(o)
#     print(idx)

# columns = ['topic_num', 'topics', 'num_articles', 'resid']

# new_data = [[15,'hello hello hello', 55, 4.4],
#             [16, 'good bye good bye', 99, 3.3]]

# new_df = pd.DataFrame(data=new_data, columns=columns)
# new_df.to_csv('topics/topics1.csv', mode='a', header=None)

# articles = pd.read_csv('sorted_articles_with_category.csv')
# topics = pd.read_csv('complete_topics_with_category.csv')

# topic_count = {}
# for _, row in topics.iterrows():
#     topic_count[row['topic_num']] = row['num_articles']

# num_articles = []
# for _, row in articles.iterrows():
#     num_articles.append(topic_count[row['topic_num']])

# articles['bubble_size'] = num_articles
# articles.to_csv('sorted_articles_with_category.csv',header=True)


bc_dailies = 'scraped_articles/BC_Dailies.csv'
national = 'scraped_articles/National.csv'
local_weeklies = 'scraped_articles/Local_Weeklies.csv'
df = pd.read_csv(bc_dailies)
# df2 = pd.read_csv(national)
# df3 = pd.read_csv(local_weeklies)

# df = pd.concat([df,df2,df3])
df.drop_duplicates(subset=['text'],keep='first',inplace=True) 
df = df[df['text'].notna()]
# df['processed_text'] = df['text'].apply(process_text)
# processed_text = df['processed_text'].tolist()
# print(processed_text[0])
# processed_text = [' '.join(x) for x in processed_text]
# print()
# print(type(processed_text[0]))
# processed_text = ' '.join(processed_text)
# print()
# print(processed_text)

old_stemmed_topics = [
    'life love don friend world ve someth load',
    'th login usernam subscrib news offic contact hi sorri order view',
    'school student district parent learn class teacher plan',
    'health case test hospit care outbreak death provinc',
    'game team play season player leagu hockey coach',
    'invest td investor financi tax advisor bank insur',
    'parti conserv tool elect govern liber mr trudeau',
    'kamloop donat week support local afford free media advertis',
    'lake park trail wildfir bc citi river area',
    'estat real hous price market galleri agent',
    'polic rcmp man investig offic court report black',
    'art festiv film artist music event theatr perform',
    'compani fish product seafood price market salmon million',
    'burnabi busi food photograph photo jennif restaur custom',
    'vehicl car drive driver engin wheel model litr'
]

stemmed_topics = [
    'life live good film world friend realli music',
    'school student class parent district teacher learn',
    'case test infect death new hospit confirm number',
    'game playoff canuck play goal shot seri score',
    'trail citi park lake mountain river hike',
    'festiv root blue salmon arm onlin',
    'polic rcmp offic man investig driver vehicl video',
    'flight affect row aug air airlin vancouv',
    'sale market juli hous price estat quarter real',
    'govern communiti program support ei worker benefit fund',
    'mask wear ferri custom store test face mandatori',
    'care senior term mackenzi long survey resid',
    'prize ticket grand bc event licenc game',
    'file photo ap elect state wednesday',
    'plant garden seed flower tomato dear helen berri'
]

unstemmed_topics = [
    'life loving don friends world ve something loads', 
    'th login username subscriber news office contact hi sorry order viewed',
    'school students district parent learning class teacher plan',
    'health case testing hospital careful outbreak deaths province', 
    'game team plays seasons players league hockey coaching',
    'investments td investors financial tax advisor banked insurance',
    'party conservation tools election government liberals mr trudeau',
    'kamloops donation week support local afford free media advertising',
    'lake parking trailing wildfire bc city rivers area',
    'estate real house prices market gallery agent',
    'police rcmp man investigation office court reported black',
    'art festivities film artist music event theatre performance',
    'company fish product seafood prices market salmon million',
    'burnaby business foods photograph photos jennifer restaurant customers',
    'vehicle car drive driver engineering wheel model litre',
]

stemmed_topics = [topic.upper() for topic in stemmed_topics]
print(stemmed_topics)