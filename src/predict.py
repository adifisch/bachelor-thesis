import torch
from tqdm import tqdm
from sklearn import metrics
from sklearn.model_selection import train_test_split
from pathlib import Path
import sys
path_root = Path(__file__).parents[2]
sys.path.append(str(path_root))
print(sys.path)
from qna_essentials import QnAClassifier, create_data_loader, eval_model, get_tokenizer, load_custom_dataset
def main():
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    PRE_TRAINED_MODEL_NAME = 'deepset/gbert-base'
    mfaq = load_custom_dataset("clips/mfaq", "de_flat", "validation")
    tokenizer = get_tokenizer(PRE_TRAINED_MODEL_NAME)
    class_names = ['nomatch', 'match']
    BATCH_SIZE = 8
    MAX_LEN = 512

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


    model = QnAClassifier(len(class_names), PRE_TRAINED_MODEL_NAME)
    print(sys.path)
    model.load_state_dict(torch.load("models\qna_classifier_2023-03-28.bin"))
    model = model.to(device)

    mfaq_dl = create_data_loader(mfaq, tokenizer, MAX_LEN, BATCH_SIZE)
    y_questions, y_answers, y_pred, y_pred_probs, y_test = get_predictions(
    model,
    mfaq_dl
    )
    print(y_pred)
    print(y_test)
    print(y_pred_probs)
    #print(y_questions)
    #print(y_answers)
    print("Precision:", metrics.precision_score(y_test, y_pred))
    print("Recall:", metrics.recall_score(y_test, y_pred))
    print("Accuracy:", metrics.accuracy_score(y_test, y_pred))
    #print(metrics.classification_report(y_test, y_pred, target_names=class_names))
    #print(metrics.confusion_matrix(y_test, y_pred, labels=class_names))
    
if __name__ == "__main__":
  main()