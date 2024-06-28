import os
from src.face_detection import FaceDetection
from src.model_evaluation import ModelEvaluation
from src.utils import create_directories, load_image, save_predictions

def main():
    # ディレクトリ設定
    dataset_path = 'data/images'
    annotation_file = 'data/annotations/annotations.csv'
    predictions_dir = 'data/predictions'
    results_dir = 'results'
    model_names = ['mtcnn', 'blazeface', 'dsfd', 'retinaface']

    create_directories([predictions_dir, results_dir])

    # 顔検出器の初期化
    detector = FaceDetection()

    # 画像ファイルの取得
    image_files = [f for f in os.listdir(dataset_path) if f.endswith(('.jpg', '.jpeg', '.png'))]

    # 各モデルで顔検出を実行し、予測結果を保存
    for model_name in model_names:
        for image_file in image_files:
            image_path = os.path.join(dataset_path, image_file)
            image = load_image(image_path)

            # 顔検出
            boxes = detector.detect_faces(image, model_name)

            # 予測結果の保存
            save_predictions(boxes, image_file.split('.')[0], model_name, predictions_dir)

    # 評価
    evaluator = ModelEvaluation(annotation_file, predictions_dir)
    evaluation_results = {}
    for model_name in model_names:
        metrics = evaluator.evaluate_model(model_name)
        evaluation_results[model_name] = metrics
        print(f"{model_name} evaluation metrics: {metrics}")

    # 結果の保存
    results_file = os.path.join(results_dir, 'evaluation_results.csv')
    with open(results_file, 'w') as f:
        f.write('model,precision,recall,f1_score\n')
        for model_name, metrics in evaluation_results.items():
            f.write(f"{model_name},{metrics['precision']},{metrics['recall']},{metrics['f1_score']}\n")

if __name__ == "__main__":
    main()
