import pandas as pd
import numpy as np

data = pd.read_csv("all.csv")
cp = pd.read_csv("cp24.csv")
record = pd.read_csv("the_record/the_record.csv")

data = pd.concat([data,cp,record], sort = False)
#drop this extra index column that was accidentally carried over
# data = data.drop('Unnamed: 0', 1)
data = data[data.columns.drop(list(data.filter(regex='Unnamed: ')))]


#the following lines are not combined into one drop_duplicate because if we try to drop with subset=["title", "text"] it would only drop rows where both the title AND text match each other
#but we want to drop all instances where the title OR the text is a duplicate

#drop articles with the same title (keeps the first entry by default)
data = data.drop_duplicates(subset="title")

#drop articles with the same text (keeps the first entry by default)
data = data.drop_duplicates(subset="text")

#convert text column and publish date to strings
data["text"] = data["text"].astype(str)
data["publish_date"] = data["publish_date"].astype(str)

#authors was stored as a full string such as "[Kathryn Hudson]" so this line will convert it into a proper list of strings such as ["Kathryn Hudson"]
data['authors'] = data.authors.apply(lambda x: x[1:-1].split(','))
#keeping only first entry since often other junk would be included in the list such as "Min. Read"
data['authors'] = data['authors'].str[0]

#this line will drop any rows that have titles which match any of the strings in the list
data = data[~data["title"].isin(["Terms of Use", "Privacy Policy", "-", "- The Weather Network", "Public Appearances"])]

#drops articles with empty body text
data = data[~data["text"].isin(["nan"])]


# only keeps rows that have an outlet which we have scraped
# MODIFY THIS LINE WHEN YOU ADD NEW OUTLETS
data = data[data["outlet"].isin(["thestar", "The Record", "cbc", "ctvnews", "nationalpost", "torontosun"])]


#drops all rows where the Authors list is empty *AND* the publish date is empty
#using an or condition because the only time this condition is false is when both predicates are false 
#which would be the case when an articles is missing authors and publish date
data = data[(data['authors'] != "") | (data['publish_date'] != "nan")]


data.sort_values(by=['outlet'])

data.to_csv("clean.csv")