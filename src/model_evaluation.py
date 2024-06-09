import os
import cv2
import numpy as np
import pandas as pd
from sklearn.metrics import precision_score, recall_score, f1_score

class ModelEvaluation:
    def __init__(self, ground_truth_file, predictions_dir, iou_threshold=0.5):
        self.ground_truth = pd.read_csv(ground_truth_file)
        self.predictions_dir = predictions_dir
        self.iou_threshold = iou_threshold

    def evaluate_model(self, model_name):
        all_true_boxes = []
        all_pred_boxes = []

        for index, row in self.ground_truth.iterrows():
            image_id = row['image_id']
            true_boxes = self.extract_true_boxes(row)
            pred_boxes = self.load_predictions(image_id, model_name)

            true_boxes, pred_boxes = self.match_boxes(true_boxes, pred_boxes)

            all_true_boxes.extend(true_boxes)
            all_pred_boxes.extend(pred_boxes)

        precision = precision_score(all_true_boxes, all_pred_boxes)
        recall = recall_score(all_true_boxes, all_pred_boxes)
        f1 = f1_score(all_true_boxes, all_pred_boxes)

        return {'precision': precision, 'recall': recall, 'f1_score': f1}

    def extract_true_boxes(self, row):
        true_boxes = []
        for i in range(row['num_faces']):
            x1 = row[f'face_{i}_x1']
            y1 = row[f'face_{i}_y1']
            x2 = row[f'face_{i}_x2']
            y2 = row[f'face_{i}_y2']
            true_boxes.append([x1, y1, x2, y2])
        return true_boxes

    def load_predictions(self, image_id, model_name):
        pred_file = os.path.join(self.predictions_dir, f'{image_id}_{model_name}.txt')
        pred_boxes = []
        if os.path.exists(pred_file):
            with open(pred_file, 'r') as file:
                lines = file.readlines()
                for line in lines:
                    x1, y1, x2, y2 = map(int, line.strip().split())
                    pred_boxes.append([x1, y1, x2, y2])
        return pred_boxes

    def calculate_iou(self, box1, box2):
        x1 = max(box1[0], box2[0])
        y1 = max(box1[1], box2[1])
        x2 = min(box1[2], box2[2])
        y2 = min(box1[3], box2[3])

        intersection = max(0, x2 - x1) * max(0, y2 - y1)
        area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
        area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
        union = area1 + area2 - intersection

        iou = intersection / union if union > 0 else 0
        return iou

    def match_boxes(self, true_boxes, pred_boxes):
        matched_true = []
        matched_pred = []

        for true_box in true_boxes:
            matched = False
            for pred_box in pred_boxes:
                iou = self.calculate_iou(true_box, pred_box)
                if iou >= self.iou_threshold:
                    matched = True
                    matched_pred.append(1)
                    break
            matched_true.append(1 if matched else 0)

        for pred_box in pred_boxes:
            matched = False
            for true_box in true_boxes:
                iou = self.calculate_iou(true_box, pred_box)
                if iou >= self.iou_threshold:
                    matched = True
                    break
            matched_pred.append(0 if not matched else 1)

        return matched_true, matched_pred

if __name__ == "__main__":
    ground_truth_file = 'data/annotations/annotations.csv'
    predictions_dir = 'data/predictions'
    evaluator = ModelEvaluation(ground_truth_file, predictions_dir)

    models = ['mtcnn', 'blazeface', 'dsfd', 'retinaface']
    for model in models:
        metrics = evaluator.evaluate_model(model)
        print(f"{model} evaluation metrics: {metrics}")
