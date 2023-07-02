# %%
import requests
from tqdm import tqdm
from pathlib import Path
from time import sleep
# %%
cfg_path = Path('cfg_files')
data_path = Path('data')/'maps'
data_path.mkdir(parents = True, exist_ok = True)
# %%
with (cfg_path/'images.txt').open('r') as f:
    url_list = f.readlines()
print(len(url_list))
for i in tqdm(range(len(url_list))):
    url = url_list[i].strip()
    file_name = data_path/(str(i) + url[-4:])

    if file_name.exists():
        continue

    sleep(1)
    r = requests.get(url)
    with open(file_name,'wb') as f:
        f.write(r.content)