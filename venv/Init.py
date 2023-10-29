import spacy
from spacy.tokens import DocBin
from tqdm import tqdm
from spacy.util import filter_spans
from spacy.scorer import Scorer
import json
import numpy as np
from spacy.training.example import Example
from sklearn.metrics import classification_report, accuracy_score, precision_score, recall_score, f1_score

model_best =  spacy.load('C:/Users/dima_/Downloads/Telegram Desktop/Test2/Test2/output/model-best')

def data_into_dataset(model ,file_data_path,file_output_path):
    db = DocBin()
    with open(file_data_path, 'r', encoding='utf-8') as f:
        TRAIN_DATA = json.load(f)
        for text, annot in tqdm(TRAIN_DATA['annotations']):
            doc = model.make_doc(text)
            ents = []
            for start, end, label in annot["entities"]:
                span = doc.char_span(start, end, label=label, alignment_mode="contract")
                if span is None:
                    print("Skipping entity")
                else:
                    ents.append(span)
            doc.ents = ents
            db.add(doc)

        db.to_disk(file_output_path)

def calculate_metrics(model_best):
    with open("C:/Users/dima_/Downloads/Telegram Desktop/Test2/Test2/data.json", 'r', encoding='utf-8') as f:
        content = f.read()
        json_data = json.loads(content)

        Precision = []
        Recall = []
        F1Score = []
        Accuracy = []

        for label in json_data["annotations"]:
            true_labels = []
            predicted_labels = []
            doc = model_best(label[0])

            for ent in doc.ents:
                predicted_labels.append(ent.label_)

            for tag in label[1]["entities"]:
                true_labels.append(tag[2])

            if len(true_labels) != len(predicted_labels):
                continue

            precision = precision_score(true_labels, predicted_labels, average='weighted', zero_division=1)
            recall = recall_score(true_labels, predicted_labels, average='weighted', zero_division=1)
            f1score = f1_score(true_labels, predicted_labels, average='weighted')
            accuracy = accuracy_score(true_labels, predicted_labels)

            Precision.append(precision)
            Recall.append(recall)
            F1Score.append(f1score)
            Accuracy.append(accuracy)

        mean_precision = np.nanmean(np.nan_to_num(Precision)) if np.any(Precision) else 0
        mean_recall = np.nanmean(np.nan_to_num(Recall)) if np.any(Recall) else 0
        mean_f1score = np.nanmean(np.nan_to_num(F1Score)) if np.any(F1Score) else 0
        mean_accuracy = np.nanmean(np.nan_to_num(Accuracy)) if np.any(Accuracy) else 0

        return {
            "mean_precision": mean_precision,
            "mean_recall": mean_recall,
            "mean_f1score": mean_f1score,
            "mean_accuracy": mean_accuracy
        }
def pridict(model, data):
    doc = model(data)

    for ent in doc.ents:
            print(ent.label_ , " - ", ent.text)


def compute_metrics(model, json_file_path):
    with open(json_file_path, 'r', encoding='utf-8') as f:
        content = f.read()
        json_data = json.loads(content)

        Precision = []
        Recall = []
        F1Score = []
        Accuracy = []

        for label in json_data["annotations"]:
            true_labels = []
            predicted_labels = []
            doc = model(label[0])

            for ent in doc.ents:
                predicted_labels.append(ent.label_)

            for tag in label[1]["entities"]:
                true_labels.append(tag[2])

            if len(true_labels) != len(predicted_labels):
                continue

            precision = precision_score(true_labels, predicted_labels, average='weighted', zero_division=1)
            recall = recall_score(true_labels, predicted_labels, average='weighted', zero_division=1)
            f1score = f1_score(true_labels, predicted_labels, average='weighted')
            accuracy = accuracy_score(true_labels, predicted_labels)

            Precision.append(precision)
            Recall.append(recall)
            F1Score.append(f1score)
            Accuracy.append(accuracy)

        mean_precision = np.nanmean(np.nan_to_num(Precision)) if np.any(Precision) else 0
        mean_recall = np.nanmean(np.nan_to_num(Recall)) if np.any(Recall) else 0
        mean_f1score = np.nanmean(np.nan_to_num(F1Score)) if np.any(F1Score) else 0
        mean_accuracy = np.nanmean(np.nan_to_num(Accuracy)) if np.any(Accuracy) else 0

        metrics = {
            "mean_precision": mean_precision,
            "mean_recall": mean_recall,
            "mean_f1score": mean_f1score,
            "mean_accuracy": mean_accuracy
        }

        return metrics

print(compute_metrics(model_best, "C:/Users/dima_/Downloads/Telegram Desktop/Test2/Test2/data.json"))

# with open("E:/Progects/Python/Test2/main_data.json", 'r', encoding='utf-8') as f:
#     content = f.read()
#     json_data = json.loads(content)
#
#     Precision = []
#     Recall = []
#     F1Score = []
#     Accuracy = []
#
#     for label in json_data["annotations"]:
#         true_labels = []
#         predicted_labels = []
#         doc = model_best(label[0])
#
#         for ent in doc.ents:
#             predicted_labels.append(ent.label_)
#
#         for tag in label[1]["entities"]:
#             true_labels.append(tag[2])
#
#         if len(true_labels) != len(predicted_labels):
#             continue
#
#         precision = precision_score(true_labels, predicted_labels, average='weighted', zero_division=1)
#         recall = recall_score(true_labels, predicted_labels, average='weighted', zero_division=1)
#         f1score = f1_score(true_labels, predicted_labels, average='weighted')
#         accuracy = accuracy_score(true_labels, predicted_labels)
#
#         print("True Labels:", true_labels)
#         print("Predicted Labels:", predicted_labels)
#
#         print("Precision:", precision)
#         Precision.append(precision)
#         print("Recall:", recall)
#         Recall.append(recall)
#         print("F1 Score:", f1score)
#         F1Score.append(f1score)
#         print("Accuracy:", accuracy)
#         Accuracy.append(accuracy)
#
#     mean_precision = np.nanmean(np.nan_to_num(Precision)) if np.any(Precision) else 0
#     mean_recall = np.nanmean(np.nan_to_num(Recall)) if np.any(Recall) else 0
#     mean_f1score = np.nanmean(np.nan_to_num(F1Score)) if np.any(F1Score) else 0
#     mean_accuracy = np.nanmean(np.nan_to_num(Accuracy)) if np.any(Accuracy) else 0
#
#     print("Mean Precision:", mean_precision)
#     print("Mean Recall:", mean_recall)
#     print("Mean F1 Score:", mean_f1score)
#     print("Mean Accuracy:", mean_accuracy)