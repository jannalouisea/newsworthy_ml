import pandas as pd
from helper import word_count, process_text

# create new dataframes by publish_date
def get_groups(file_name):
    df = pd.read_csv(file_name)
    df.drop_duplicates(subset=['url'],keep='first',inplace=True)
    # df = df.sort_values(by=['publish_date'])

    df['word_count'] = df['text'].apply(word_count)
    df['processed_text'] = df['text'].apply(process_text)

    by_date = df.groupby('publish_date')
    groups = [by_date.get_group(x) for x in by_date.groups]
    return groups

