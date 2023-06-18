# %%
import json
from pathlib import Path
import pandas as pd
# %%
cfg_path = Path('cfg_files')
data_path = Path('data')
data_path.mkdir(parents = True, exist_ok = True)
# %%
combined_list = []
# %%
dataset1_path = cfg_path/'dataset_twitter-scraper_2023-06-03_23-07-08-886.csv'
df = pd.read_csv(dataset1_path)
combined_list += df['media/0/media_url'].to_list()
# %%
dataset2_path = cfg_path/'manual_scrape.json'
with dataset2_path.open('r') as f:
    dataset2 = json.load(f)
combined_list += list(dataset2.keys())
# %%
def dupliDeleter(l):
    l_fixed = []
    for i in range(len(l)):
        url = l[i]
        if '?' in url:
            url = url[0:url.index('?')]
            url += '.jpg'
        l_fixed.append(url)
    return l_fixed
combined_list = dupliDeleter(combined_list)
# %%
combined_list = list(set(combined_list))
print(len(combined_list))
# %%
combined_list_path = cfg_path/'images.txt'
with combined_list_path.open('w') as f:
    for url in combined_list:
        f.write(url + '\n')
        