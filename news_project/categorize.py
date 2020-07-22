import guidedlda
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer

from helper import process_text

texts = [
    ['water', 'new york', 'film', 'hospit', 'nova scotia'],
    ['trudeau', 'minist', 'program', 'govern', 'parti', 'prime', 'public'],
    ['mask', 'open', 'peopl', 'health', 'servic', 'covid', 'public', 'distanc'],
    ['case', 'covid', 'health', 'new', 'test', 'number', 'death', 'provinc'],
    ['peopl', 'just', 'work', 'famili', 'make', 'want', 'day', 'home'],
    ['lake', 'fish', 'speci', 'river', 'agricultur', 'wildlif', 'salmon', 'fisheri'],
    ['north', 'road', 'south', 'dog', 'highway', 'area', 'river', 'citi'],
    ['compani', 'cent', 'home', 'sale', 'million', 'price', 'tax', 'month'],
    ['communiti', 'ridg', 'bc', 'mapl', 'citi', 'park', 'facebook'],
    ['langley', 'com', 'artist', 'festiv', 'www', 'perform', 'award', 'facebook'],
    ['car', 'vehicl', 'drive', 'driver', 'engin', 'wheel', 'new', 'seat'],
    ['govern', 'canada', 'pandem', 'covid', 'work', 'canadian', 'busi', 'peopl'],
    ['juli', 'et', 'brault', 'hydro', 'wednesday', 'morn', 'secondari', 'tank', 'burger'],
    ['team', 'player', 'play', 'game', 'season', 'toronto', 'leagu', 'sport'],
    ['polic', 'offic', 'rcmp', 'investig', 'surrey', 'man', 'juli', 'report']
]

seed_topic_list = [
    ['Game', 'team', 'player', 'season','sport','league','hockey','baseball','coach'],
    ['coronavirus','covid','mask','infection','cases','test','death','vaccine'],
    ['government', 'case', 'police', 'officer','trudeau','minister','politics','ottawa','canada'],
    ['health','medicine','doctor','pharmacy','medication','pill','insurance'],
    ['music','movie','actor','theatre','art','entertainment','painting','festival'],
    ['science','tech','code','solar','electric'],
    ['crisis','humanitarian','aid'],
    ['living','food','restaurant','bar','burger','pizza','eat','drink','activity'],
    ['company','work','business','office','salary']
]

# -politics, -covid, -health, -science+tech, -sports, -arts+entertainment, -crisis updates, -lifestyle, business

words = [item for sublist in texts for item in sublist]
words = list(dict.fromkeys(words))
# {word : word_id, ...}

seed_topic_list_process = []
# process_text
for l in seed_topic_list:
    l_processed = []
    for x in l:
        x = process_text(x)
        if x in words:
            l_processed.append(x)
    l_processed = [item for sublist in l_processed for item in sublist]
    seed_topic_list_process.append(l_processed)
seed_topic_list = seed_topic_list_process




vectorizer = CountVectorizer(
    max_df=100,
    min_df=0,
    max_features=1000,
    ngram_range=(1,2),
    preprocessor=' '.join
)

X = vectorizer.fit_transform(texts)
X = X.toarray()

words = vectorizer.get_feature_names()
word2id = dict((v, idx) for idx, v in enumerate(words))


model = guidedlda.GuidedLDA(
    n_topics=9,
    n_iter=100,
    random_state=7,
    refresh=20
)

# seed the topics
seed_topics = {}
for t_id, st in enumerate(seed_topic_list):
  for word in st:
    # seed_topics[word_id] = seed_topic_number
    seed_topics[word2id[word]] = t_id

model.fit(X, seed_topics=seed_topics, seed_confidence=0.35)

n_top_words = 5
# num_topics x num_words
topic_word = model.topic_word_
print(type(topic_word))

for i, topic_dist in enumerate(topic_word):
    # iterate through topics with prob of word belonging to topic

    all_words = np.array(words)
    # sorted by prob, entry is word_id
    sorted_topics = np.argsort(topic_dist)
    sorted_topics = all_words[sorted_topics]
    sorted_topics = sorted_topics[:-(n_top_words+1):-1]
    # topic_words = all_words[np.argsort(topic_dist)][:-(n_top_words+1):-1]
    print('topic: {}: {}'.format(i, ' '.join(sorted_topics)))
