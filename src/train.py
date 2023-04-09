import os
import torch
import pandas as pd
import numpy as np
from transformers import AutoModel, BertTokenizer
from transformers import Trainer, TrainingArguments, HfArgumentParser
from datasets import Dataset
from torch.utils.data import DataLoader

RANDOM_SEED = 142
np.random.seed(RANDOM_SEED)
torch.manual_seed(RANDOM_SEED)
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
PRE_TRAINED_MODEL_NAME = 'deepset/gbert-base'
model = AutoModel.from_pretrained(PRE_TRAINED_MODEL_NAME)
tokenizer = BertTokenizer.from_pretrained(
    PRE_TRAINED_MODEL_NAME,
    use_fast=True
    )
data = pd.read_csv('../data/faq_info_labels.csv')
ds = Dataset.from_pandas(data)
ds = ds.with_format("torch")
dataloader = DataLoader(ds, batch_size=32, num_workers=4, shuffle=True)

for batch in dataloader:
    print(batch)