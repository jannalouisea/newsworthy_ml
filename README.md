# AI4good_news

## Scraping data

_Steps to Run_

1. Install all requirements listed in requirements.txt
> `pip3 install -r requirements.txt`

2. Ensure you're running python 3
> `python3 --version` or `which python3`

3. Once all dependencies are installed, run the script
> `python3 scrape_data.py`
> Or just `python` depending on how you have your aliases set up.

4. This should output a file called `all_news.csv`. Then run the cleaning script
> `python clean_data.py`

5. This should output a new file called `clean.csv` which will contain all working data entries for the GTA and BC
