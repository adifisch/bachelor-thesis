import torch
import numpy as np
import pandas as pd
from torch import nn
from tqdm import tqdm
import datasets
from datasets import load_dataset
from torch.utils.data import Dataset, DataLoader
from transformers import BertModel, BertTokenizer
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def load_custom_dataset(dataset, options, split):
    ds = load_dataset(dataset, options)
    df = ds.data[split]
    df_q = pd.DataFrame.from_dict(df['question'])
    df_a = pd.DataFrame.from_dict(df['answer'])
    
    df_q = df_q.rename(columns={0: "question"})
    df_a = df_a.rename(columns={0: "answer"})
    frames = [df_q, df_a]
    
    df = pd.concat(frames,axis=1, join="inner")
    half_len = int(len(df)/2)
    data_matches = df.iloc[:half_len,:]
    data_distinct = df.iloc[half_len:,:]
    data_distinct = data_distinct.reset_index(drop=True)
    data_matches = data_matches.reset_index(drop=True)
    df1 = data_distinct.iloc[np.random.permutation(data_distinct.index)].reset_index(drop=True)
    data_distinct['answer'] = df1['answer']
    data_distinct['matching'] = 0
    data_matches['matching'] = 1
    df = pd.concat([data_matches,data_distinct])
    df = df.sample(frac=1).reset_index(drop=True)
    return df

def get_tokenizer(model_name):
    return BertTokenizer.from_pretrained(model_name)

class QnAClassifier(nn.Module):
    def __init__(self, n_classes, model_name):
        super(QnAClassifier, self).__init__()
        self.bert = BertModel.from_pretrained(model_name)
        self.drop = nn.Dropout(p=0.3)
        self.out = nn.Linear(self.bert.config.hidden_size, n_classes)
    def forward(self, input_ids, attention_mask):
        _, pooled_output = self.bert(
            input_ids=input_ids,
            attention_mask=attention_mask,
            return_dict=False
        )
        output = self.drop(pooled_output)
        return self.out(output)
    
class QnADataset(Dataset):
    def __init__(self, question, answer, targets, tokenizer, max_length):
        self.question = question
        self.answer = answer
        self.targets = targets
        self.tokenizer = tokenizer
        self.max_length = int(max_length)
    
    def __len__(self):
        return len(self.answer)

    def __getitem__(self, item):
        question = str(self.question[item])
        context = str(self.answer[item])
        target = self.targets[item]
        
        encoding = self.tokenizer(
            question,
            context,
            max_length = self.max_length,
            add_special_tokens = True,
            padding='max_length',
            truncation='only_second',
            return_attention_mask = True,
            return_token_type_ids = False,
            return_tensors = 'pt'
        )
        return {
            'question_text': question,
            'answer_text': context,
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'targets': torch.tensor(target, dtype=torch.long)
        }

def collate(batch):
        #print(batch)
        return batch.to(device, non_blocking=True)
        #input_nodes, _, mfgs = batch
        #return input_nodes, [block.to(device, non_blocking=True) for block in mfgs]
    
def create_data_loader(df, tokenizer, max_length, batch_size):
    if isinstance(df, datasets.Dataset):
        ds = QnADataset(
            question=df['question'],
            answer=df['answer'],
            targets=df['matching'],
            tokenizer=tokenizer,
            max_length=max_length
        )
    else:
        #question=df.question.to_numpy(),
        #answer=df.answer.to_numpy(),
        #targets=df.matching.to_numpy(),
        ds = QnADataset(
            question=df.question.to_numpy(),
            answer=df.answer.to_numpy(),
            targets=df.matching.to_numpy(),
            tokenizer=tokenizer,
            max_length=max_length
        )
    return DataLoader(
        ds,
        batch_size = batch_size,
        shuffle=True,
        num_workers=2
        #,collate_fn=collate
    )

def train_epoch(
    model,
    data_loader,
    loss_fn,
    optimizer,
    device,
    scheduler,
    n_examples
):
    model = model.train()
    losses = []
    correct_predictions = 0
    for d in tqdm(data_loader):
        input_ids = d["input_ids"].to(device)
        attention_mask = d["attention_mask"].to(device)
        targets = d["targets"].to(device)
        outputs = model(
            input_ids=input_ids,
            attention_mask=attention_mask
        )
        _, preds = torch.max(outputs, dim=1)
        loss = loss_fn(outputs, targets)
        correct_predictions += torch.sum(preds == targets)
        losses.append(loss.item())
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        scheduler.step()
        optimizer.zero_grad()
    return correct_predictions.double() / n_examples, np.mean(losses)

def eval_model(model, data_loader, loss_fn, device, n_examples):
    model = model.eval()
    losses = []
    correct_predictions = 0
    with torch.no_grad():
        for d in tqdm(data_loader):
            input_ids = d["input_ids"].to(device)
            attention_mask = d["attention_mask"].to(device)
            targets = d["targets"].to(device)
            
            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask
            )
            _, preds = torch.max(outputs, dim=1)
            loss = loss_fn(outputs, targets)
            correct_predictions += torch.sum(preds == targets)
            losses.append(loss.item())
    return correct_predictions.double() / n_examples, np.mean(losses)
