import sys
args = sys.argv[1:]
import torch
import numpy as np
import pandas as pd
from torch import nn
from collections import defaultdict
from sklearn.model_selection import train_test_split
from transformers import AdamW, get_linear_schedule_with_warmup
from pathlib import Path
from qna_essentials import QnAClassifier, create_data_loader, train_epoch, eval_model, get_tokenizer
path_root = Path(__file__).parents[2]
sys.path.append(str(path_root))

TRAINING_PARAMS = {}
TRAINING_PARAMS['BATCHSIZE'] = 8
TRAINING_PARAMS['EPOCHS'] = 10
TRAINING_PARAMS['MAXLEN'] = 512
TRAINING_PARAMS['SEED'] = 42
TRAINING_PARAMS['MODEL'] = 'deepset/gbert-base'
TRAINING_PARAMS['DATAURL'] = 'data/bert_dataset.csv'
class_names = ['nomatch', 'match']

for i in TRAINING_PARAMS.keys():
  try:
    print('-'+i.lower())
    if i == "MODEL":
      model_name = args[args.index('-'+i.lower()) + 1]
      if model_name == 'bert':
        TRAINING_PARAMS['MODEL'] = 'deepset/gbert-base'
        TRAINING_PARAMS['DATAURL'] = 'data/bert_dataset.csv'
      elif model_name == 't5':
        TRAINING_PARAMS['MODEL'] = 'dehio/german-qg-t5-quad'
        TRAINING_PARAMS['DATAURL'] = 'data/t5_dataset.csv'
    else:
      args.index('-'+i.lower())
  except:
    continue
print('Training Parameters are:',TRAINING_PARAMS)

np.random.seed(TRAINING_PARAMS['SEED'])
torch.manual_seed(TRAINING_PARAMS['SEED'])
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
qna_data = pd.read_csv(TRAINING_PARAMS['DATAURL'])
tokenizer = get_tokenizer(TRAINING_PARAMS['MODEL'])

def main():
  df_train, df_test = train_test_split(
    qna_data,
    test_size=0.1,
    random_state=TRAINING_PARAMS['SEED']
  )
  df_val, df_test = train_test_split(
    df_test,
    test_size=0.5,
    random_state=TRAINING_PARAMS['SEED']
  )

  train_data_loader = create_data_loader(df_train, tokenizer, TRAINING_PARAMS['MAXLEN'], TRAINING_PARAMS['BATCHSIZE'])
  val_data_loader = create_data_loader(df_val, tokenizer, TRAINING_PARAMS['MAXLEN'], TRAINING_PARAMS['BATCHSIZE'])
  test_data_loader = create_data_loader(df_test, tokenizer, TRAINING_PARAMS['MAXLEN'], TRAINING_PARAMS['BATCHSIZE'])


  model = QnAClassifier(len(class_names), TRAINING_PARAMS['MODEL'])
  model = model.to(device)
  optimizer = AdamW(model.parameters(), lr=2e-5, correct_bias=False)
  total_steps = len(train_data_loader) * TRAINING_PARAMS['EPOCHS']
  scheduler = get_linear_schedule_with_warmup(
    optimizer,
    num_warmup_steps=0,
    num_training_steps=total_steps
  )
  loss_fn = nn.CrossEntropyLoss().to(device)

  history = defaultdict(list)
  best_accuracy = 0
  for epoch in range(TRAINING_PARAMS['EPOCHS']):
      print(f'Epoch {epoch + 1}/{TRAINING_PARAMS["EPOCHS"]}')
      print('-' * 10)
      train_acc, train_loss = train_epoch(
          model,
          train_data_loader,
          loss_fn,
          optimizer,
          device,
          scheduler,
          len(df_train)
      )
      print(f'Train loss {train_loss} accuracy {train_acc}')
      val_acc, val_loss = eval_model(
          model,
          val_data_loader,
          loss_fn,
          device,
          len(df_val)
      )
      print(f'Val   loss {val_loss} accuracy {val_acc}')
      print()
      history['train_acc'].append(train_acc)
      history['train_loss'].append(train_loss)
      history['val_acc'].append(val_acc)
      history['val_loss'].append(val_loss)
      if val_acc > best_accuracy:
          torch.save(model.state_dict(), 'best_model_state.bin')
          best_accuracy = val_acc
          
if __name__ == "__main__":
  main()