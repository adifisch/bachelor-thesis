import torch
from tqdm import tqdm
import datasets
from sklearn import metrics
from sklearn.model_selection import train_test_split
from qna_essentials import QnAClassifier, create_data_loader, eval_model, get_tokenizer, load_custom_dataset
import matplotlib.pyplot as plt
import seaborn as sns

def main():
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    PRE_TRAINED_MODEL_NAME = 'deepset/gbert-base'
    mfaq = load_custom_dataset("clips/mfaq", "de_flat", "validation")
    test_data = datasets.load_from_disk('data/bert_bobby_dataset')['test']
    test_data = test_data.remove_columns(['name', 'sourceName', 'sourceUrl', 'faqLabels'])
    tokenizer = get_tokenizer(PRE_TRAINED_MODEL_NAME)
    class_names = ['nomatch', 'match']
    BATCH_SIZE = 8
    MAX_LEN = 512
    
    model = QnAClassifier(len(class_names), PRE_TRAINED_MODEL_NAME)
    model.load_state_dict(torch.load("models/qna_classifier_2023-10-05.bin"))
    model = model.to(device)

    def get_predictions(model, data_loader):
        model = model.eval()
        questions = []
        answers = []
        predictions = []
        prediction_probs = []
        real_values = []
        with torch.no_grad():
            for d in tqdm(data_loader):
                question_text = d["question_text"]
                answer_text = d["answer_text"]
                input_ids = d["input_ids"].to(device)
                attention_mask = d["attention_mask"].to(device)
                targets = d["targets"].to(device)
                outputs = model(
                    input_ids=input_ids,
                    attention_mask=attention_mask
                )
                _, preds = torch.max(outputs, dim=1)
                questions.extend(question_text)
                answers.extend(answer_text)
                predictions.extend(preds)
                prediction_probs.extend(outputs)
                real_values.extend(targets)
        predictions = torch.stack(predictions).cpu()
        prediction_probs = torch.stack(prediction_probs).cpu()
        real_values = torch.stack(real_values).cpu()
        return questions, answers, predictions, prediction_probs, real_values
    
    test_dl = create_data_loader(test_data, tokenizer, MAX_LEN, BATCH_SIZE)
    mfaq_dl = create_data_loader(mfaq, tokenizer, MAX_LEN, BATCH_SIZE)
    
    #y_questions, y_answers, y_pred, y_pred_probs, y_test = get_predictions(model,test_dl)
    y_questions, y_answers, y_pred, y_pred_probs, y_test = get_predictions(model,mfaq_dl)
    
    wrong_preds = open('appendix/wrong_preds_mfaq.txt','w')
    for i in range(len(y_pred)):
        if y_pred[i] != y_test[i]:
            wrong_preds.write(y_questions[i] + '\t' + y_answers[i] +'\n')
            
    print("Precision:", metrics.precision_score(y_test, y_pred))
    print("Recall:", metrics.recall_score(y_test, y_pred))
    print("Accuracy:", metrics.accuracy_score(y_test, y_pred))
    print(metrics.classification_report(y_test, y_pred, target_names=class_names))
    print(metrics.confusion_matrix(y_test, y_pred))
    confusion_matrix = metrics.confusion_matrix(y_test, y_pred)
    
    ax = plt.subplot()
    sns.set(font_scale=2.0) 
    sns.heatmap(confusion_matrix, annot=True, ax=ax, cmap="Blues", fmt="g", cbar=False);  

    # Labels, title and ticks
    label_font = {'size':'16'} 
    ax.set_xlabel('Vorausgesagte Labels ', fontdict=label_font);
    ax.set_ylabel('Echte Labels', fontdict=label_font);

    title_font = {'size':'16'}
    ax.set_title('Konfusionsmatrix clips/mfaq-Validation', fontdict=title_font);
    #ax.set_title('Konfusionsmatrix Bobbi-Test', fontdict=title_font);

    ax.tick_params(axis='both', which='major', labelsize=14) 
    ax.xaxis.set_ticklabels(['Kein Match', 'Match']);
    ax.yaxis.set_ticklabels(['Kein Match', 'Match']);
    plt.show()
        
if __name__ == "__main__":
  main()
  