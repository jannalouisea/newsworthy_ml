# newsworthy.ml

`newsworthy.ml` is a platform that was built during the project development phase of the 2020 AI4Good Lab in Montreal, Canada. It is a map-based web app that encourages users to explore different news topics from various geographical locations, instead of relying solely on social media platforms for news, which tend to be sensational and "click-baity".

Our goal, as a team of six, was to tackle the problem of information literacy in the digital age and aim to increase the exposure of readers to less visible news outlets and topics. The code in this repository contributes to one component of the platform that we developed in a span of three weeks. We aim to build out various other features in the following months.


### To view the live deployment

You can visit our live website [here](https://reclassifed-frontend-git-master.harpriyabagri.vercel.app/)


## Data Scraping

_Steps to Run_

1. Install all requirements listed in requirements.txt
> `pip3 install -r requirements.txt`

2. Ensure you're running python 3
> `python3 --version` or `which python3`

3. Once all dependencies are installed, run the script.<br> **Warning: this script takes ~ 2 hours to run.**
> `python3 scrape_data.py`
> Or just `python` depending on how you have your aliases set up.

4. This should output a file called `all_news.csv`. Then run the cleaning script
> `python clean_data.py`

5. This should output a new file called `clean.csv` which will contain all working data entries for the GTA and BC

6. This file should then be run through the topic modelling script which will output a dataset containing topics in a file called `sorted_articles.csv`. It will also output a dataset of topics called `complete_topics.csv`. The topics will be categorized and unstemmmed. 
> `python nmf.py`

7. This dataset is then run through a location-finding script which outputs `final_df.csv`
> `python geocode_articles.ipynb`

8. After the locations are added to the `final_df.csv` this file is then put through a geojson conversion script called `geojson_convert.py` which outputs an `articles.json` file which is then passed to the front end via an S3 bucket.
> `python geojson_convert.py`
