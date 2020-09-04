import pandas as pd
from predict import predict_category
from helper import process_text, get_unstemmed_word, casual_tokenizer

bc_dailies = 'scraped_articles/BC_Dailies.csv'
national = 'scraped_articles/National.csv'
local_weeklies = 'scraped_articles/Local_Weeklies.csv'
df = pd.read_csv(bc_dailies)
df2 = pd.read_csv(national)
df3 = pd.read_csv(local_weeklies)

df = pd.concat([df,df2,df3])
df.drop_duplicates(subset=['text'],keep='first',inplace=True) 
df = df[df['text'].notna()]
'''
# df['processed_text'] = df['text'].apply(process_text)
# processed_text = df['processed_text'].tolist()
# print(processed_text[0])
# processed_text = [' '.join(x) for x in processed_text]
# print()
# print(type(processed_text[0]))
# processed_text = ' '.join(processed_text)
# print()
# print(processed_text)
'''

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

# stemmed_topics = [topic.upper() for topic in stemmed_topics]
# print(stemmed_topics)

texts = df['text'].tolist()
# print(type(text))
# print(type(text[0]))
# print(text[0])

for doc in texts:
    temp = doc.split(' ')
    for word in temp:
        ve = process_text(word)
        if len(ve) > 0:
            if ve[0] == 'hi':
                print(word)
                print('---------------------------------')
            if ve[0] == 'mr':
                print(word)
                print('---------------------------------')
            elif ve[0] == 'td':
                print(word)
                print('---------------------------------')
            # elif word == 'hi':
            #     print('WORD: ',word)
            #     print(doc)
            #     print('---------------------------------')
            elif word.lower() == 'mr':
                print('WORD: ',word)
                print(doc)
                print('---------------------------------')
            elif word.lower() == 'td':
                print('WORD: ',word)
                print(doc)
                print('---------------------------------')
            elif word.lower() == 've':
                print('WORD: ',word)
                print(doc)
                print('---------------------------------')
          
            
    