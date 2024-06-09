import os
import pandas as pd
from sklearn.metrics import precision_score, recall_score, f1_score

class Evaluation:
    def __init__(self, ground_truth_path, predictions_path):
        self.ground_truth = pd.read_csv(ground_truth_path)
        self.predictions = pd.read_csv(predictions_path)

    def evaluate(self):
        # 予測結果と実際のラベルの比較
        y_true = self.ground_truth['label'].tolist()
        y_pred = self.predictions['predicted_label'].tolist()

        # 評価指標の計算
        precision = precision_score(y_true, y_pred, average='macro')
        recall = recall_score(y_true, y_pred, average='macro')
        f1 = f1_score(y_true, y_pred, average='macro')

        return precision, recall, f1

    def print_results(self, precision, recall, f1):
        print("Evaluation Results:")
        print(f"Precision: {precision:.2f}")
        print(f"Recall: {recall:.2f}")
        print(f"F1 Score: {f1:.2f}")

if __name__ == "__main__":
    ground_truth_path = 'data/ground_truth.csv'
    predictions_path = 'data/predictions.csv'

    evaluator = Evaluation(ground_truth_path, predictions_path)
    precision, recall, f1 = evaluator.evaluate()
    evaluator.print_results(precision, recall, f1)
