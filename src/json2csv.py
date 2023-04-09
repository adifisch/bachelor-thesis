# %%
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tqdm as tqdm

# %%
from os import walk

# %%
import os.path
from os import path

# %%
target_dirs = ['../data/unpack-faq/json', '../data/unpack-faq/home/ssds/start-faq-refresh/json']
files = []

# %%
for dataDir in target_dirs:
    for (root, dirnames, filenames) in walk(dataDir):
        for file in filenames:
            files.append(dataDir+file)

# %%
import json

# %%
file_infos = []

# %%
for dataDir in target_dirs:
    for (root, dirnames, filenames) in walk(dataDir):
        for file in filenames:
            files.append(dataDir + '/' + file)

# %%
for file in files:
    info = open(file, mode='r',encoding="utf8")
    file_infos.append(info.read())
    info.close()

# %%
test = map(json.loads, file_infos)

# %%
for file in files:
    info = open(file, mode='r',encoding="utf8")
    file_infos.append(info.read())
    info.close()

# %%
len(file_infos)

# %%
file_infos = list(map(json.loads, file_infos))

# %%
file_infos_df = pd.DataFrame(file_infos)
file_infos_df.head(n=2)

# %%
file_infos_df.to_csv('../data/faq_info.csv', index=None, header=True)


