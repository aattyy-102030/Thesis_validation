import os
import pandas as pd
from sklearn.metrics import confusion_matrix, classification_report

class Evaluation:
    def __init__(self, ground_truth_path, predictions_path):
        self.ground_truth = pd.read_csv(ground_truth_path)
        self.predictions = pd.read_csv(predictions_path)

    def evaluate(self):
        # 予測結果と実際のラベルの比較
        y_true = self.ground_truth['label'].values
        y_pred = self.predictions['predicted_label'].values

        # 混同行列と分類レポートの生成
        cm = confusion_matrix(y_true, y_pred)
        report = classification_report(y_true, y_pred)

        return cm, report

    def print_results(self, confusion_matrix, classification_report):
        print("Confusion Matrix:")
        print(confusion_matrix)
        print("\nClassification Report:")
        print(classification_report)

if __name__ == "__main__":
    ground_truth_path = 'data/ground_truth.csv'
    predictions_path = 'data/predictions.csv'

    evaluator = Evaluation(ground_truth_path, predictions_path)
    cm, report = evaluator.evaluate()
    evaluator.print_results(cm, report)
