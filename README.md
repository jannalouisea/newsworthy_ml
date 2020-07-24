# AI4good_news

## Scraping data

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

6. This file should then be run through the topic modelling script which will output a dataset containing topics in a file called `sorted_articles.csv`. It will also output a dataset of topics called `complete_topics.csv`.
> `python nmf.py`

7. The dataset containing just the topics will be run through the categorizing modelling script which will output the topics with the greater news category that they belong to. 
> `python predict.py`

8. This dataset is then run through a location finding script which outputs `final_df.csv`
> `python` *janna add your script name*

9. After the locations are added to the `final_df.csv` this file is then put through a geojson conversion script called `geojson_convert.py` which outputs an `articles.json` file which is then passed to the front end via an S3 bucket.
> `python geojson_convert.py`
