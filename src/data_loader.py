import os
import pandas as pd
import cv2
from tqdm import tqdm

class DataLoader:
    def __init__(self, image_dir, annotation_file, output_dir):
        self.image_dir = image_dir
        self.annotation_file = annotation_file
        self.output_dir = output_dir
        self.annotations = pd.read_csv(annotation_file)
        
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

    def process_images(self):
        for idx, row in tqdm(self.annotations.iterrows(), total=len(self.annotations)):
            image_path = os.path.join(self.image_dir, row['image_id'])
            image = cv2.imread(image_path)
            
            # ここで必要な前処理を行います
            if image is not None:
                face_boxes = self.extract_faces(row)
                self.save_processed_image(image, face_boxes, row['image_id'])

    def extract_faces(self, row):
        face_boxes = []
        for i in range(row['num_faces']):
            x1 = row[f'face_{i}_x1']
            y1 = row[f'face_{i}_y1']
            x2 = row[f'face_{i}_x2']
            y2 = row[f'face_{i}_y2']
            face_boxes.append((x1, y1, x2, y2))
        return face_boxes

    def save_processed_image(self, image, face_boxes, image_id):
        for i, (x1, y1, x2, y2) in enumerate(face_boxes):
            face = image[y1:y2, x1:x2]
            face_path = os.path.join(self.output_dir, f'{image_id}_face_{i}.jpg')
            cv2.imwrite(face_path, face)

if __name__ == "__main__":
    image_dir = 'data/images'
    annotation_file = 'data/annotations/annotations.csv'
    output_dir = 'data/processed_images'
    
    data_loader = DataLoader(image_dir, annotation_file, output_dir)
    data_loader.process_images()
