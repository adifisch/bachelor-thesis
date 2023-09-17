import torch
import numpy as np
import pandas as pd
from io import StringIO
from html.parser import HTMLParser

class MLStripper(HTMLParser):
    def __init__(self):
        super().__init__()
        self.reset()
        self.strict = False
        self.convert_charrefs= True
        self.text = StringIO()
    def handle_data(self, d):
        self.text.write(d)
    def get_data(self):
        return self.text.getvalue()

def strip_tags(html):
    s = MLStripper()
    s.feed(html)
    return s.get_data()

def preprocess_data(dataset, name):
    dataset = pd.read_csv(dataset)
    dataset = dataset.drop(4328)
    dataset = dataset.drop(2591)
    dataset = dataset.drop(6019)

    dataset['question'] = dataset['question'].map(lambda x: strip_tags(x))
    dataset['answer'] = dataset['answer'].map(lambda x: strip_tags(x))


    dataset['question'] = dataset['question'].str.replace('\(*Stand:*\s*\d+\.\d+\.\d+(,\s*\d*:*\d*\s*\w*)*\)+', '', regex=True);
    dataset['answer'] = dataset['answer'].str.replace('\(*Stand:*\s*\d+\.\d+\.\d+(,\s*\d*:*\d*\s*\w*)*\)+', '', regex=True);
    dataset['question'] = dataset['question'].str.replace('\(*Stand:*\s*\d+\.\d+\.\d+(,\s*\d*\.*\d*\s*\w*)*\)+', '', regex=True);
    dataset['answer'] = dataset['answer'].str.replace('\(*Stand:*\s*\d+\.\d+\.\d+(,\s*\d*\.*\d*\s*\w*)*\)+', '', regex=True);
    dataset['question'] = dataset['question'].str.replace('\(*Stand:*\s*\d+\.\d+\.\d+(,\s*\d*:*\d*\s*\w*)*\)*', '', regex=True);
    dataset['answer'] = dataset['answer'].str.replace('\(*Stand:*\s*\d+\.\d+\.\d+(,\s*\d*:*\d*\s*\w*)*\)*', '', regex=True);
    dataset['question'] = dataset['question'].str.replace('\(*Stand:*\s*\d+\.\d+\.\d+(,\s*\d*\.*\d*\s*\w*)*\)*', '', regex=True);
    dataset['answer'] = dataset['answer'].str.replace('\(*Stand:*\s*\d+\.\d+\.\d+(,\s*\d*\.*\d*\s*\w*)*\)*', '', regex=True);

    dataset['question'] = dataset['question'].str.replace('\(*Stand:\s*\d+\.*\s*\w*\.*\s*\d+(,\s*\d+:\d+\s*\w*)*\)*', '', regex=True);
    dataset['answer'] = dataset['answer'].str.replace('\(*Stand:\s*\d+\.*\s*\w*\.*\s*\d+(,\s*\d+:\d+\s*\w*)*\)*', '', regex=True);

    dataset['question'] = dataset['question'].str.replace('\(*Frage\s*und\s*Antwort\s*angepasst\s*am\s*\d+\.*\s*\w*\.*\s*\d+(,\s*\d+:\d+\s*\w*)*\)*', '', regex=True);
    dataset['answer'] = dataset['answer'].str.replace('\(*Frage\s*und\s*Antwort\s*angepasst\s*am\s*\d+\.*\s*\w*\.*\s*\d+(,\s*\d+:\d+\s*\w*)*\)*', '', regex=True);

    dataset['question'] = dataset['question'].str.replace('\(*Frage\s*angepasst\s*am\s*\d+\.*\s*\w*\.*\s*\d+(,\s*\d+:\d+\s*\w*)*\)*', '', regex=True);
    dataset['answer'] = dataset['answer'].str.replace('\(*Frage\s*angepasst\s*am\s*\d+\.*\s*\w*\.*\s*\d+(,\s*\d+:\d+\s*\w*)*\)*', '', regex=True);

    dataset['question'] = dataset['question'].str.replace('\(*Neue\s*Frage\s*vom\s*\d+\.*\s*\w*\.*\s*\d+(,\s*\d+:\d+\s*\w*)*\)*', '', regex=True);
    dataset['answer'] = dataset['answer'].str.replace('\(*Neue\s*Frage\s*vom\s*\d+\.*\s*\w*\.*\s*\d+(,\s*\d+:\d+\s*\w*)*\)*', '', regex=True);

    dataset['question'] = dataset['question'].str.replace('\(*Antwort\s*angepasst\s*am\s*\d+\.*\s*\w*\.*\s*\d+(,\s*\d+:\d+\s*\w*)*\)*', '', regex=True);
    dataset['answer'] = dataset['answer'].str.replace('\(*Antwort\s*angepasst\s*am\s*\d+\.*\s*\w*\.*\s*\d+(,\s*\d+:\d+\s*\w*)*\)*', '', regex=True);

    dataset['question'] = dataset['question'].str.replace('\(*Antwort\s*ergänzt\s*am\s*\d+\.*\s*\w*\.*\s*\d+(,\s*\d+:\d+\s*\w*)*\)*', '', regex=True);
    dataset['answer'] = dataset['answer'].str.replace('\(*Antwort\s*ergänzt\s*am\s*\d+\.*\s*\w*\.*\s*\d+(,\s*\d+:\d+\s*\w*)*\)*', '', regex=True);

    dataset['question'] = dataset['question'].str.replace('\(+\d+\.*\s*\w*\.*\s*\d+(,\s*\d+:\d+\s*\w*)*\)+', '', regex=True);
    dataset['answer'] = dataset['answer'].str.replace('\(+\d+\.*\s*\w*\.*\s*\d+(,\s*\d+:\d+\s*\w*)*\)+', '', regex=True);
    dataset['question'] = dataset['question'].str.replace('\(+\d+\.*\s*\d*\.*\s*\d+(,\s*\d+:\d+\s*\w*)*\)+', '', regex=True);
    dataset['answer'] = dataset['answer'].str.replace('\(+\d+\.*\s*\d*\.*\s*\d+(,\s*\d+:\d+\s*\w*)*\)+', '', regex=True);

    dataset['question'] = dataset['question'].str.replace('\(+\d+\.*\s*\w*\.*\s*\d+(,\s*\d+\.*\d+\s*\w*)*\)+', '', regex=True);
    dataset['answer'] = dataset['answer'].str.replace('\(+\d+\.*\s*\w*\.*\s*\d+(,\s*\d+\.*\d+\s*\w*)*\)+', '', regex=True);
    dataset['question'] = dataset['question'].str.replace('\(+\d+\.*\s*\d*\.*\s*\d+(,\s*\d+\.*\d+\s*\w*)*\)+', '', regex=True);
    dataset['answer'] = dataset['answer'].str.replace('\(+\d+\.*\s*\d*\.*\s*\d+(,\s*\d+\.*\d+\s*\w*)*\)+', '', regex=True);

    dataset.to_csv('data/'+name+'.csv', index=None, header=True)
    
preprocess_data('data/faq_info_labels.csv','bert_dataset')
preprocess_data('data/faq_info.csv','t5_dataset')