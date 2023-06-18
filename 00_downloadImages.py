# %%
import requests
from tqdm import tqdm
from pathlib import Path
# %%
cfg_path = Path('cfg_files')
data_path = Path('data')
data_path.mkdir(parents = True, exist_ok = True)
# %%
with (cfg_path/'images.txt').open('r') as f:
    url_list = f.readlines()

for i in tqdm(range(len(url_list))):
    url = url_list[i].strip()
    r = requests.get(url)
    with open((data_path/(str(i) + url[-4:])),'wb') as f:
        f.write(r.content)